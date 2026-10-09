"""Docker sandbox for the Code tab's agent (driven through the docker CLI; no SDK dependency).

One long-lived container runs `sleep infinity`; everything else is a `docker exec` into it. Files
live only inside the container (/workspace), so `reset()` is a clean slate. Nothing here knows
about any language or framework:

- commands run in a process registry (Codex unified-exec style): each gets an id `pN`, its own
  process group and a log file; a command that is still running after the yield time keeps
  running in the background and can be polled or killed;
- listening ports are discovered from /proc/net/tcp, and a relay inside the container forwards
  the published host port to whichever container port the preview should show;
- checkpoints are commits to a shadow git repo outside /workspace.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

DOCKERFILE_DIR = Path(__file__).resolve().parent.parent / "sandbox"
IMAGE = "bonsai-code-sandbox:v3"  # bump when the image layout changes (old images lack /opt/bonsai)
NAME = "bonsai-code"
WORKDIR = "/workspace"
PREVIEW_PORT = 5180  # host port; the in-container relay forwards it to the preview target port
RELAY_PORT = 8100  # container side of the published port
STATE = "/tmp/bonsai"  # harness state inside the container (never in the user's workspace)
IGNORED_DIRS = ("node_modules", ".git", "dist", "build", ".venv", "__pycache__", ".bonsai")
MAX_PROCESSES = 16  # background processes at once (cheap; eviction beyond this is reported)
PROTECT_NEW_S = 60  # a process started this recently is not evicted
# Resources: by default the container gets this share of the host's RAM (override with
# BONSAI_SANDBOX_MEMORY_GB / BONSAI_SANDBOX_CPUS). Docker Desktop's VM is the hard ceiling.
MEMORY_SHARE = 0.5
CPU_SHARE = 2 / 3  # the rest is left to the host (and the model server)
VM_HEADROOM_GB = 1.0  # left to the Docker VM itself
# A plain, non-interactive terminal for model commands (the variables Codex's unified exec sets).
# Tool-specific settings (npm, pip, …) belong in the image's environment, not here.
PLAIN_ENV = "NO_COLOR=1 FORCE_COLOR=0 TERM=dumb COLORTERM= PAGER=cat GIT_PAGER=cat GH_PAGER=cat LANG=C.UTF-8 LC_ALL=C.UTF-8"

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07|\r(?!\n)")


def plain(text: str) -> str:
    """Drop terminal escape codes and carriage-return redraws: they cost tokens and confuse the model."""
    return _ANSI.sub("", text)


class SandboxError(RuntimeError):
    pass


def _docker(*args: str, input: bytes | None = None, timeout: float | None = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], input=input, capture_output=True, timeout=timeout)


@dataclass
class Process:
    id: str
    command: str
    started: float
    pgid: int = 0
    cursor: int = 0  # bytes of the log already returned to the model
    watch_cursor: int = 0  # bytes the harness has scanned for errors (separate, so the model still sees them)
    exit_code: int | None = None
    ports: set[int] = field(default_factory=set)
    reported_ports: set[int] = field(default_factory=set)
    autostart: bool = False
    oom: bool = False  # killed by the kernel for exceeding the container's memory limit


# Lists listening TCP ports with the process group that owns each (no iproute2 needed).
_PORTS_PY = r"""
import os, glob
inodes = {}
for f in ("/proc/net/tcp", "/proc/net/tcp6"):
    try:
        for line in open(f).read().splitlines()[1:]:
            p = line.split()
            if p[3] == "0A":
                inodes[p[9]] = int(p[1].rsplit(":", 1)[1], 16)
    except OSError:
        pass
seen = set()
for fd in glob.glob("/proc/[0-9]*/fd/*"):
    try:
        t = os.readlink(fd)
    except OSError:
        continue
    if t.startswith("socket:["):
        port = inodes.get(t[8:-1])
        if port is not None:
            pid = fd.split("/")[2]
            try:
                pgid = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[3])
            except (OSError, IndexError, ValueError):
                pgid = 0
            if (port, pgid) not in seen:
                seen.add((port, pgid))
                print(port, pgid)
"""


class Sandbox:
    def __init__(self, name: str = NAME, image: str = IMAGE, port: int = PREVIEW_PORT):
        self.name, self.image, self.port = name, image, port
        self.procs: dict[str, Process] = {}
        self._next = 1
        self._lock = threading.Lock()
        self.preview_target: int | None = None  # container port the relay forwards to
        self.limits: dict = {}  # resources of the running container (set by start())
        self.evicted: list[tuple[str, str]] = []  # processes stopped to make room (reported, then cleared)

    # ------------------------------------------------------------ lifecycle
    def docker_ok(self) -> bool:
        try:
            return _docker("info", "--format", "{{.ServerVersion}}", timeout=10).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def image_ok(self) -> bool:
        return _docker("image", "inspect", self.image, timeout=20).returncode == 0

    def running(self) -> bool:
        r = _docker("inspect", "-f", "{{.State.Running}}", self.name, timeout=20)
        return r.returncode == 0 and r.stdout.strip() == b"true"

    def build_image(self, log: Callable[[str], None] = print):
        p = subprocess.Popen(
            ["docker", "build", "--progress=plain", "-t", self.image, str(DOCKERFILE_DIR)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        for line in p.stdout:
            log(line.rstrip("\n"))
        if p.wait() != 0:
            raise SandboxError(f"docker build failed (exit {p.returncode})")

    def ensure(self, log: Callable[[str], None] = print):
        """Image built and container running (builds the image on first use)."""
        if not self.docker_ok():
            raise SandboxError("Docker is not running (start Docker Desktop)")
        if not self.image_ok():
            log(f"Building image {self.image} (first run only, takes a few minutes)…")
            self.build_image(log)
        if not self.running():
            log("Starting sandbox container…")
            self.start()
            if self.limits.get("warning"):
                log("WARNING: " + self.limits["warning"])
            log(f"Sandbox resources: {self.limits.get('memory_gb')} GB RAM, {self.limits.get('cpus')} CPUs")

    def resources(self) -> tuple[float, int, str | None]:
        """(memory GB, CPUs, warning) for the container: MEMORY_SHARE of host RAM unless overridden,
        capped by what Docker Desktop's VM actually has."""
        host_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
        want = float(os.environ.get("BONSAI_SANDBOX_MEMORY_GB") or host_gb * MEMORY_SHARE)
        cpus = int(os.environ.get("BONSAI_SANDBOX_CPUS") or max(1, int((os.cpu_count() or 4) * CPU_SHARE)))
        warning = None
        r = _docker("info", "--format", "{{.MemTotal}} {{.NCPU}}", timeout=20)
        try:
            vm_bytes, vm_cpus = r.stdout.decode().split()
            vm_gb = int(vm_bytes) / 2**30
            if want > vm_gb - VM_HEADROOM_GB:
                warning = (f"Sandbox wants {want:.0f} GB RAM but Docker Desktop's VM has only {vm_gb:.1f} GB; using "
                           f"{vm_gb - VM_HEADROOM_GB:.1f} GB. Raise it in Docker Desktop → Settings → Resources → Memory.")
                want = vm_gb - VM_HEADROOM_GB
            cpus = min(cpus, int(vm_cpus))
        except ValueError:
            warning = f"could not read Docker's VM size ({r.stderr.decode().strip()}); using {want:.0f} GB"
        return round(want, 1), cpus, warning

    def start(self):
        self.remove()
        mem_gb, cpus, warning = self.resources()
        self.limits = {"memory_gb": mem_gb, "cpus": cpus, "warning": warning}
        r = _docker(
            "run", "-d", "--init", "--name", self.name,
            "-p", f"127.0.0.1:{self.port}:{RELAY_PORT}",
            # Pinned cores (not a --cpus quota): nproc, Python and PyTorch then see the real count and
            # don't start a thread per host core.
            "--memory", f"{int(mem_gb * 1024)}m", "--cpuset-cpus", f"0-{cpus - 1}",
            "-w", WORKDIR, self.image, "sleep", "infinity",
        )
        if r.returncode != 0:
            raise SandboxError(r.stderr.decode(errors="replace").strip())
        with self._lock:
            self.procs.clear()
            self._next = 1
        self.preview_target = None

    def remove(self):
        _docker("rm", "-f", self.name, timeout=60)

    def reset(self):
        self.kill_all()
        self.start()

    # ------------------------------------------------------------ harness-internal commands
    def sh(self, cmd: str, timeout: float = 60, input: bytes | None = None) -> tuple[int, str]:
        """A short command for the harness itself (hooks, checks, listings): not in the registry."""
        # The timeout runs inside the container: killing only the docker CLI would leave the command running.
        t = max(1, int(timeout))
        args = (["exec"] + (["-i"] if input is not None else [])
                + ["-w", WORKDIR, self.name, "timeout", "-k", "5", str(t), "bash", "-c", cmd])
        try:
            r = _docker(*args, input=input, timeout=t + 15)
        except subprocess.TimeoutExpired:
            return 124, f"[timed out after {t}s]"
        out = plain((r.stdout + r.stderr).decode(errors="replace"))
        return r.returncode, (out + f"\n[timed out after {t}s]" if r.returncode == 124 else out)

    # ------------------------------------------------------------ process registry (model commands)
    def spawn(self, command: str, autostart: bool = False) -> Process:
        """Start `command` detached in its own process group, logging to STATE/proc/<id>.log."""
        live = [p for p in self.procs.values() if p.exit_code is None]
        if len(live) >= MAX_PROCESSES:
            self.refresh_exits()  # finished processes the harness hasn't polled yet don't count
        with self._lock:
            live = [p for p in self.procs.values() if p.exit_code is None]
            if len(live) >= MAX_PROCESSES:
                # Make room: the oldest process that serves no port and isn't brand new. Never silently.
                now = time.time()
                idle = ([p for p in live if not p.ports and now - p.started > PROTECT_NEW_S]
                        or [p for p in live if not p.ports] or live)
                victim = min(idle, key=lambda p: p.started)
                self.kill(victim.id)
                self.evicted.append((victim.id, victim.command))
            pid = f"p{self._next}"
            self._next += 1
            proc = Process(pid, command, time.time(), autostart=autostart)
            self.procs[pid] = proc
        d = f"{STATE}/proc"
        inner = (f"echo $$ > {d}/{pid}.pid; cd {WORKDIR}; export {PLAIN_ENV}; bash -o pipefail -lc {shlex.quote(command)} < /dev/null "
                 f"> {d}/{pid}.log 2>&1; echo $? > {d}/{pid}.exit")
        r = _docker("exec", "-d", self.name, "bash", "-c", f"mkdir -p {d} && exec setsid bash -c {shlex.quote(inner)}")
        if r.returncode != 0:
            proc.exit_code = 126
            raise SandboxError(r.stderr.decode(errors="replace").strip() or "could not start the command")
        for _ in range(40):  # the pid file appears almost immediately
            code, out = self.sh(f"cat {d}/{pid}.pid 2>/dev/null", timeout=10)
            if out.strip().isdigit():
                proc.pgid = int(out.strip())
                break
            time.sleep(0.05)
        return proc

    def poll(self, pid: str, watch: bool = False) -> tuple[str, bool]:
        """New output since the last read (the model's cursor, or the harness's with watch=True), and
        whether the process has exited."""
        proc = self.procs[pid]
        d = f"{STATE}/proc"
        cur = proc.watch_cursor if watch else proc.cursor
        # Exit status first (so output written before exit is never missed), then size, then the new bytes.
        code, out = self.sh(
            f"ex=$(cat {d}/{pid}.exit 2>/dev/null); sz=$(wc -c < {d}/{pid}.log 2>/dev/null || echo 0); "
            f"printf '%s %s\\n' \"${{ex:-none}}\" \"$sz\"; tail -c +{cur + 1} {d}/{pid}.log 2>/dev/null "
            f"| head -c $(( ${{sz:-0}} - {cur} ))", timeout=20)
        head, _, body = out.partition("\n")
        ex, _, size = head.strip().partition(" ")
        if size.strip().isdigit():
            if watch:
                proc.watch_cursor = max(proc.watch_cursor, int(size.strip()))
            else:
                proc.cursor = max(proc.cursor, int(size.strip()))
        if ex.lstrip("-").isdigit() and proc.exit_code is None:
            proc.exit_code = int(ex)
            if proc.exit_code == 137 and self.oom_kills_changed():
                proc.oom = True
            if proc.pgid:  # background children (`cmd &`) would otherwise outlive the process untracked
                self.sh(f"kill -TERM -{proc.pgid} 2>/dev/null; true", timeout=10)
        return body, proc.exit_code is not None

    def wait(self, pid: str, seconds: float, cancel: threading.Event | None = None,
             idle: float | None = None) -> tuple[str, bool]:
        """Collect output until the process exits or `seconds` pass (then it keeps running). With `idle`,
        also return once the process has been silent that long."""
        chunks, deadline, delay = [], time.monotonic() + seconds, 0.05
        last_output = time.monotonic()
        while True:
            out, done = self.poll(pid)
            chunks.append(out)
            if out:
                last_output = time.monotonic()
            if done or time.monotonic() >= deadline or (idle and time.monotonic() - last_output >= idle):
                return "".join(chunks), done
            if cancel is not None and cancel.is_set():
                self.kill(pid)
                out, _ = self.poll(pid)
                return "".join(chunks) + out, True
            time.sleep(delay)
            delay = min(delay * 1.6, 0.5)

    def oom_kills_changed(self) -> bool:
        """True if the container's out-of-memory kill counter went up since the last check."""
        code, out = self.sh("grep '^oom_kill ' /sys/fs/cgroup/memory.events | cut -d' ' -f2", timeout=10)
        n = int(out.strip()) if out.strip().isdigit() else 0
        changed, self._oom_seen = n > getattr(self, "_oom_seen", 0), n
        return changed

    def refresh_exits(self):
        """Mark processes whose exit file exists (one listing for all of them)."""
        code, out = self.sh(f"cd {STATE}/proc 2>/dev/null && for f in *.exit; do echo \"$f $(cat $f)\"; done", timeout=20)
        for line in out.splitlines():
            name, _, val = line.partition(" ")
            pid = name.removesuffix(".exit")
            if pid in self.procs and self.procs[pid].exit_code is None and val.strip().lstrip("-").isdigit():
                self.procs[pid].exit_code = int(val.strip())

    def kill(self, pid: str) -> bool:
        proc = self.procs.get(pid)
        if proc is None:
            return False
        if proc.exit_code is None and proc.pgid:
            self.sh(f"kill -TERM -{proc.pgid} 2>/dev/null; sleep 0.3; kill -KILL -{proc.pgid} 2>/dev/null; true",
                    timeout=20)
        if proc.exit_code is None:
            proc.exit_code = -9
        if self.preview_target in proc.ports:
            self.preview_target = None
        return True

    def kill_all(self, include_autostart: bool = True):
        for pid, p in list(self.procs.items()):
            if p.exit_code is None and (include_autostart or not p.autostart):
                self.kill(pid)

    def running_procs(self) -> list[Process]:
        return [p for p in self.procs.values() if p.exit_code is None]

    # ------------------------------------------------------------ ports and preview
    def scan_ports(self, mark: bool = True) -> list[tuple[Process, int]]:
        """Update each process's listening ports; returns newly opened (process, port) pairs. With
        mark=False the ports are refreshed but stay "new" for the harness's own event processing."""
        code, out = self.sh("python3 -", input=_PORTS_PY.encode(), timeout=20)
        by_pgid = {p.pgid: p for p in self.procs.values() if p.pgid and p.exit_code is None}
        current: dict[str, set[int]] = {pid: set() for pid in self.procs}
        for line in out.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                port, pgid = int(parts[0]), int(parts[1])
                if pgid in by_pgid and port != RELAY_PORT:
                    current[by_pgid[pgid].id].add(port)
        opened = []
        for pid, ports in current.items():
            proc = self.procs[pid]
            proc.ports = ports
            for port in sorted(ports - proc.reported_ports):
                opened.append((proc, port))
                if mark:
                    proc.reported_ports.add(port)
        return opened

    _RELAY_PY = r"""
import asyncio, sys
target = int(sys.argv[1])
async def pipe(r, w):
    try:
        while data := await r.read(65536):
            w.write(data)
            await w.drain()
    except Exception:
        pass
    finally:
        w.close()
async def handle(cr, cw):
    try:
        ur, uw = await asyncio.open_connection("127.0.0.1", target)
    except OSError:
        cw.close()
        return
    await asyncio.gather(pipe(cr, uw), pipe(ur, cw))
async def main():
    server = await asyncio.start_server(handle, "0.0.0.0", int(sys.argv[2]), reuse_address=True)
    async with server:
        await server.serve_forever()
asyncio.run(main())
"""

    def set_preview(self, port: int):
        """Point the published host port at `port` inside the container (a raw TCP relay, so websockets
        such as hot reload work unchanged)."""
        if self.preview_target == port and self._relay_alive():
            return
        pidf = f"{STATE}/relay.pid"
        # Stop the old relay and wait until its port is free, then start the new one.
        self.sh(f"[ -f {pidf} ] && kill $(cat {pidf}) 2>/dev/null; for i in $(seq 20); do "
                f"[ -f {pidf} ] && kill -0 $(cat {pidf}) 2>/dev/null || break; sleep 0.1; done; rm -f {pidf}", timeout=10)
        self.write(f"{STATE}/relay.py", self._RELAY_PY)
        _docker("exec", "-d", self.name, "bash", "-c",
                f"echo $$ > {pidf}; exec python3 {STATE}/relay.py {port} {RELAY_PORT} > {STATE}/relay.log 2>&1")
        self.preview_target = port
        for _ in range(20):
            if self._relay_alive():
                break
            time.sleep(0.1)

    def _relay_alive(self) -> bool:
        pidf = f"{STATE}/relay.pid"
        return self.sh(f"[ -f {pidf} ] && kill -0 $(cat {pidf}) 2>/dev/null", timeout=10)[0] == 0

    def preview_up(self) -> bool:
        if self.preview_target is None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/", timeout=1) as r:
                return r.status < 500
        except Exception:
            return False

    # ------------------------------------------------------------ files
    def read(self, path: str) -> bytes:
        r = _docker("exec", "-w", WORKDIR, self.name, "cat", "--", path)
        if r.returncode != 0:
            raise SandboxError(r.stderr.decode(errors="replace").strip() or f"cannot read {path}")
        return r.stdout

    def write(self, path: str, data: str | bytes):
        data = data.encode() if isinstance(data, str) else data
        script = 'mkdir -p -- "$(dirname -- "$1")" && cat > "$1"'
        r = _docker("exec", "-i", "-w", WORKDIR, self.name, "sh", "-c", script, "sh", path, input=data)
        if r.returncode != 0:
            raise SandboxError(r.stderr.decode(errors="replace").strip() or f"cannot write {path}")

    _PEEK_PY = r"""
import bz2, gzip, lzma, os, sys, zipfile
p = sys.argv[1]
size = os.path.getsize(p)
def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024: return f"{n:.0f} {u}"
        n /= 1024
    return f"{n:.1f} TB"
openers = {".bz2": bz2.open, ".gz": gzip.open, ".xz": lzma.open, ".lzma": lzma.open}
ext = os.path.splitext(p)[1].lower()
print(f"size: {human(size)}")
try:
    if ext == ".zip":
        with zipfile.ZipFile(p) as z:
            names = z.namelist()
            print(f"zip archive, {len(names)} entries: " + ", ".join(names[:10]))
        sys.exit()
    with (openers[ext](p, "rb") if ext in openers else open(p, "rb")) as f:
        head = f.read(3000)
    if ext in openers:
        print(f"compressed ({ext[1:]}); first decompressed bytes:")
    if b"\0" in head[:1000]:
        print("binary data; first bytes: " + head[:64].hex(" "))
    else:
        print("\n".join(head.decode("utf-8", "replace").splitlines()[:25])[:2500])
except Exception as e:
    print(f"could not peek: {type(e).__name__}: {e}")
"""

    def peek(self, path: str) -> str:
        """Size, type and the first lines of a data file (decompressing gz/bz2/xz on the fly)."""
        code, out = self.sh(f"python3 - {shlex.quote(path)}", input=self._PEEK_PY.encode(), timeout=30)
        return out.strip()

    def sizes(self, paths: list[str]) -> dict[str, int]:
        if not paths:
            return {}
        code, out = self.sh("stat -c '%s %n' -- " + " ".join(shlex.quote(p) for p in paths) + " 2>/dev/null", timeout=20)
        res = {}
        for line in out.splitlines():
            n, _, name = line.partition(" ")
            if n.isdigit():
                res[name] = int(n)
        return res

    def exists(self, path: str) -> bool:
        return _docker("exec", "-w", WORKDIR, self.name, "test", "-e", path, timeout=20).returncode == 0

    def tree(self, path: str = ".", limit: int = 2000) -> list[str]:
        """Files under `path` (relative to /workspace), newest first, skipping IGNORED_DIRS."""
        prune = " -o ".join(f"-name {shlex.quote(d)}" for d in IGNORED_DIRS)
        cmd = (f"find {shlex.quote(path)} \\( {prune} \\) -prune -o -type f -printf '%T@ %p\\n' 2>/dev/null "
               f"| sort -rn | head -n {limit} | cut -d' ' -f2- | sed 's|^\\./||'")
        code, out = self.sh(cmd, timeout=30)
        return [l for l in out.splitlines() if l and not l.startswith("find:")]

    def archive(self) -> bytes:
        r = _docker("exec", self.name, "tar", "czf", "-", *[f"--exclude={d}" for d in IGNORED_DIRS if d != ".git"],
                    "-C", WORKDIR, ".", timeout=120)
        if r.returncode != 0:
            raise SandboxError(r.stderr.decode(errors="replace").strip())
        return r.stdout

    # ------------------------------------------------------------ checkpoints (shadow git, outside /workspace)
    _GIT = "git --git-dir=/ckpt --work-tree=/workspace -c user.name=bonsai -c user.email=bonsai@local"

    def checkpoint(self, message: str) -> str | None:
        """Commit the workspace to the shadow repo; returns the commit id."""
        excl = "\\n".join(f"{d}/" for d in IGNORED_DIRS if d != ".git")
        cmd = (f"[ -d /ckpt ] || {{ git init -q --bare /ckpt && printf '{excl}\\n' > /ckpt/info/exclude; }}; "
               f"{self._GIT} add -A && {self._GIT} commit -q --allow-empty -m {shlex.quote(message)} >/dev/null 2>&1; "
               f"{self._GIT} rev-parse --short HEAD")
        code, out = self.sh(cmd, timeout=60)
        last = out.strip().splitlines()[-1] if out.strip() else ""
        return last if code == 0 and re.fullmatch(r"[0-9a-f]{4,40}", last) else None

    def dirty(self) -> list[str]:
        """Files changed since the last checkpoint by anything (tools, shell commands, installs)."""
        code, out = self.sh(f"[ -d /ckpt ] && {self._GIT} status --porcelain --untracked-files=all 2>/dev/null", timeout=30)
        files = []
        for line in out.splitlines():
            if len(line) > 3:
                path = line[3:].split(" -> ")[-1].strip().strip('"')
                files.append(path)
        return files

    def checkpoints(self) -> list[dict]:
        code, out = self.sh(f"{self._GIT} log --format='%h%x09%ct%x09%s' -n 200 2>/dev/null", timeout=30)
        rows = []
        for line in out.splitlines():
            parts = line.split("\t", 2)
            if len(parts) == 3 and parts[1].isdigit():
                rows.append({"id": parts[0], "time": int(parts[1]), "message": parts[2]})
        return rows

    def restore(self, commit: str):
        if not re.fullmatch(r"[0-9a-f]{4,40}", commit):
            raise SandboxError("bad checkpoint id")
        # read-tree --reset -u makes index and files match the commit exactly (files added later go away).
        code, out = self.sh(f"{self._GIT} add -A && {self._GIT} read-tree --reset -u {commit} && {self._GIT} clean -fdq && "
                            f"{self._GIT} commit -q --allow-empty -m 'restored {commit}' >/dev/null 2>&1; true", timeout=60)
        if code != 0:
            raise SandboxError(out.strip())

    def diff(self, a: str, b: str = "HEAD") -> str:
        for c in (a, b):
            if not re.fullmatch(r"[0-9a-f]{4,40}|HEAD", c):
                raise SandboxError("bad checkpoint id")
        return self.sh(f"{self._GIT} diff --stat --patch {a} {b} | head -c 200000", timeout=60)[1]
