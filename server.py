#!/usr/bin/env python3
"""Single-step readout server.   uv run server.py   (then open http://localhost:8000)

POST /api/reset   {prompt, mode: chat|raw, thinking, reasoning_effort, system, image_b64?}  -> new session (no forward)
POST /api/step    {strategy: greedy|sample, temperature, top_p, top_k}     -> ONE forward + readouts,
                                                                           appends the chosen token
POST /api/replace {token_id}                                               -> swap the last chosen token
GET  /api/last.npz                                                         -> full logits/pre/post (float32)
GET  /api/image/{n}/info                                                   -> how image n became tokens
GET  /api/image/{n}.png                                                    -> image n as the model saw it (resized)

Free chat (second tab):
POST /api/chat/new  {system, thinking, reasoning_effort, temperature, max_tokens} -> new conversation
POST /api/chat/send {text, image_b64?}  -> text/event-stream of {"type": "delta"|"done"|"error", ...}
POST /api/chat/continue {max_tokens?}   -> resume a reply cut off by max_tokens / Stop (same stream)
POST /api/chat/stop                     -> cancel the running reply

Coding agent (third tab), tools run in a Docker sandbox (bonsai_lens/sandbox.py):
GET  /api/code/status                   -> docker/image/container/preview state, context size, plan
POST /api/code/start                    -> text/event-stream: build image (first run) + start container
POST /api/code/send {text, thinking?, reasoning_effort?, temperature?} -> start the agent (runs server-side);
                                           while it runs, the text is queued as steering instead
GET  /api/code/events?since=N           -> event-stream of the session's event log from seq N (replay + live)
POST /api/code/stop                     -> cancel the agent (and its running command)
POST /api/code/reset                    -> recreate the container, forget the conversation
GET  /api/code/tree | /api/code/file?path= | /api/code/archive  -> workspace files
GET  /api/code/transcript               -> the agent's full context, decoded
GET  /api/code/image?path= | /api/code/checkpoints | /api/code/diff?a=&b=, POST /api/code/restore {id}
"""

import asyncio
import atexit
import base64
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from bonsai_lens.agent import MAX_CONTEXT, CodeAgent
from bonsai_lens.conversation import Conversation
from bonsai_lens.sandbox import Sandbox, SandboxError
from bonsai_lens.engine import Engine, summarize

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "web" / "dist"

# MLX streams are thread-bound: load and run everything on this one thread.
mlx_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
state: dict = {"engine": None, "session": None, "last": None, "loading_error": None, "step": 0}
lock = threading.Lock()
chat: dict = {"conv": None, "cancel": threading.Event(), "busy": False}
code: dict = {"agent": None, "box": Sandbox(), "cancel": threading.Event(), "busy": False, "starting": False,
              "events": [], "epoch": 0}  # events: the session's full event log (replayed by /api/code/events)


def code_emit(ev: dict):
    """Append to the Code session's event log (thread-safe enough: list.append is atomic)."""
    log = code["events"]
    log.append({**ev, "seq": len(log), "t": time.time() * 1000})
UPLOADS = Path(tempfile.mkdtemp(prefix="bonsai-uploads-"))
atexit.register(shutil.rmtree, UPLOADS, ignore_errors=True)


def save_image(b64: str) -> str:
    from PIL import Image

    img = Image.open(io.BytesIO(base64.b64decode(b64.split(",", 1)[-1]))).convert("RGB")
    f = tempfile.NamedTemporaryFile(suffix=".png", dir=UPLOADS, delete=False)
    path = f.name
    f.close()  # Windows cannot reopen a NamedTemporaryFile while it is open.
    img.save(path)
    return path


def run(fn, *a):
    return mlx_thread.submit(fn, *a).result()


def busy(*, except_: str = "") -> str | None:
    """Why the model is unavailable to another tab (the views share one model and one MLX thread)."""
    if chat["busy"] and except_ != "chat":
        return "the Chat tab is generating; wait or stop it"
    if code["busy"] and except_ != "code":
        return "the Code agent is running; wait or stop it"
    return None


def sse_response(target, cancel: threading.Event | None = None) -> StreamingResponse:
    """Run `target(put)` on a background thread and stream every put(event) as SSE.
    A client disconnect sets `cancel`."""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def put(ev):
        loop.call_soon_threadsafe(q.put_nowait, ev)

    def work():
        try:
            target(put)
        except Exception as exc:
            put({"type": "error", "message": str(exc) if isinstance(exc, SandboxError) else repr(exc)})
        finally:
            put(None)

    threading.Thread(target=work, daemon=True).start()

    async def stream():
        try:
            while (ev := await q.get()) is not None:
                yield f"data: {json.dumps(ev)}\n\n"
        finally:  # client went away
            if cancel is not None:
                cancel.set()

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


def _load():
    try:
        t0 = time.perf_counter()
        state["engine"] = Engine()
        state["load_seconds"] = time.perf_counter() - t0
    except Exception as exc:  # surfaced via /api/info
        state["loading_error"] = repr(exc)
        raise


app = FastAPI()


class ResetReq(BaseModel):
    prompt: str
    mode: str = "chat"  # chat | raw
    thinking: bool = False
    reasoning_effort: str = "xhigh"  # xhigh | medium | low (medium adds no system text)
    system: str | None = None
    image_b64: str | None = None  # data URL or bare base64


class StepReq(BaseModel):
    strategy: str = "greedy"  # greedy | sample
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    k: int = 50  # number of candidates to return


class ReplaceReq(BaseModel):
    token_id: int


def engine() -> Engine:
    e = state["engine"]
    if e is None:
        raise HTTPException(503, state["loading_error"] or "model is still loading")
    return e


def tok_view(e: Engine, ids):
    return [{"id": int(i), "piece": e.piece(i), "text": e.decode([i])} for i in ids]


@app.get("/api/info")
def info():
    e = state["engine"]
    if e is None:
        return {"ready": False, "error": state["loading_error"]}
    tc = e.config["text_config"]
    return {
        "ready": True,
        **getattr(e, "runtime_info", {"backend": "mlx"}),
        "load_seconds": state.get("load_seconds"),
        "hidden_size": e.hidden_size,
        "vocab_size": e.vocab_size,
        "num_layers": tc["num_hidden_layers"],
        "eos_ids": sorted(e.eos_ids),
    }


@app.post("/api/reset")
def reset(req: ResetReq):
    e = engine()
    if getattr(e, "runtime_info", {}).get("lens_readouts", True) is False:
        raise HTTPException(501, "This native server does not expose Lens readouts; build the Prism readout bridge")
    image_path = save_image(req.image_b64) if req.image_b64 else None
    if req.mode == "chat":
        content = req.prompt
        if image_path:
            content = [{"type": "image"}, {"type": "text", "text": req.prompt}]
        msgs = ([{"role": "system", "content": req.system}] if req.system else []) + [
            {"role": "user", "content": content}
        ]
        text = e.render(msgs, thinking=req.thinking, reasoning_effort=req.reasoning_effort)
    else:
        text = req.prompt
        if image_path and "<|image_pad|>" not in text:
            text = "<|vision_start|><|image_pad|><|vision_end|>" + text

    with lock:
        state["session"] = run(e.new_session, text, [image_path] if image_path else None)
        state["last"] = None
        state["step"] = 0
        s = state["session"]
    return {"rendered": text, "tokens": tok_view(e, s.tokens), "prompt_len": len(s.tokens)}


@app.post("/api/step")
def step(req: StepReq):
    e = engine()
    with lock:
        s = state["session"]
        if s is None:
            raise HTTPException(400, "call /api/reset first")
        if msg := busy():
            raise HTTPException(409, msg)
        if s.tokens and s.fed < len(s.tokens) and s.tokens[-1] in e.eos_ids:
            raise HTTPException(409, "generation ended at EOS; pick another candidate or reset")
        t0 = time.perf_counter()
        n_fed = len(s.tokens) - s.fed
        readout = run(e.forward, s)
        dt = time.perf_counter() - t0
        greedy = req.strategy == "greedy"
        chosen = run(
            e.sample, readout.logits, 0.0 if greedy else req.temperature, req.top_p, req.top_k
        )
        summary = run(summarize, readout, req.k)
        position = len(s.tokens)  # index the chosen token will occupy
        s.tokens.append(chosen)
        state["last"] = readout
        state["step"] += 1
    summary["top"] = tok_view(e, summary.pop("top_ids"))
    chosen_logit = float(np.asarray(readout.logits)[chosen])
    return {
        "step": state["step"],
        "position": position,
        "fed_tokens": n_fed,
        "forward_ms": dt * 1000,
        "chosen": {**tok_view(e, [chosen])[0], "logit": chosen_logit},
        "is_eos": chosen in e.eos_ids,
        "readout": summary,
    }


@app.post("/api/replace")
def replace(req: ReplaceReq):
    """The chosen token isn't fed until the next step, so swapping it is free."""
    e = engine()
    with lock:
        s = state["session"]
        if s is None or s.fed >= len(s.tokens):
            raise HTTPException(400, "no unfed token to replace")
        s.tokens[-1] = req.token_id
    return {"chosen": tok_view(e, [req.token_id])[0], "is_eos": req.token_id in e.eos_ids}


def _lens_image(n: int) -> dict:
    e = engine()
    s = state["session"]
    if s is None or not s.image_kwargs:
        raise HTTPException(404, "current Lens session has no image")
    if msg := busy():
        raise HTTPException(409, msg)
    with lock:
        infos = run(e.image_details, s)
    if not 0 <= n < len(infos):
        raise HTTPException(404, "no such image")
    return infos[n]


@app.get("/api/image/{n}/info")
def image_info(n: int):
    return _lens_image(n)


@app.get("/api/image/{n}.png")
def image_png(n: int):
    info = _lens_image(n)
    from PIL import Image

    img = Image.open(info["source"]).convert("RGB").resize((info["width"], info["height"]), Image.Resampling.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


@app.get("/api/last.npz")
def last_npz():
    r = state["last"]
    if r is None:
        raise HTTPException(404, "no step yet")
    buf = io.BytesIO()
    np.savez(buf, logits=np.asarray(r.logits), pre_norm=np.asarray(r.pre_norm), post_norm=np.asarray(r.post_norm))
    return Response(buf.getvalue(), media_type="application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="readout_step{state["step"]}.npz"'})


# ------------------------------------------------------------------ free chat
class ChatNewReq(BaseModel):
    system: str | None = None
    thinking: bool = False
    reasoning_effort: str = "medium"
    temperature: float = 1.0
    max_tokens: int = 32768


class ChatSendReq(BaseModel):
    text: str
    image_b64: str | None = None
    thinking: bool | None = None
    reasoning_effort: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None  # per-reply output cap


class ChatContinueReq(BaseModel):
    max_tokens: int | None = None


@app.post("/api/chat/new")
def chat_new(req: ChatNewReq):
    e = engine()
    if chat["busy"]:
        raise HTTPException(409, "a reply is still streaming; stop it first")
    chat["conv"] = Conversation(
        e,
        system=req.system or None,
        thinking=req.thinking,
        reasoning_effort=req.reasoning_effort,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )
    return {"ok": True}


@app.post("/api/chat/stop")
def chat_stop():
    chat["cancel"].set()
    return {"ok": True}


@app.post("/api/chat/send")
async def chat_send(req: ChatSendReq):
    e = engine()
    _chat_check_free()
    if chat["conv"] is None:
        chat["conv"] = Conversation(e)
    conv: Conversation = chat["conv"]
    for k in ("thinking", "reasoning_effort", "temperature", "max_tokens"):
        if getattr(req, k) is not None:
            setattr(conv, k, getattr(req, k))
    image = save_image(req.image_b64) if req.image_b64 else None
    return _chat_stream(conv, lambda cancel: conv.stream(req.text, image=image, cancel=cancel))


@app.post("/api/chat/continue")
async def chat_continue(req: ChatContinueReq):
    engine()
    _chat_check_free()
    conv: Conversation | None = chat["conv"]
    if conv is None or conv.open_reply is None:
        raise HTTPException(409, "the last reply is complete; nothing to continue")
    if req.max_tokens is not None:
        conv.max_tokens = req.max_tokens
    return _chat_stream(conv, lambda cancel: conv.resume(cancel=cancel))


def _chat_check_free():
    if chat["busy"]:
        raise HTTPException(409, "a reply is already streaming")
    if msg := busy(except_="chat"):
        raise HTTPException(409, msg)


def _chat_stream(conv: Conversation, events) -> StreamingResponse:
    """Run `events(cancel)` on the MLX thread and relay its events as text/event-stream."""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    cancel = chat["cancel"]
    cancel.clear()
    chat["busy"] = True

    def work():  # runs on the MLX thread
        try:
            q_put({"type": "start", "thinking": conv.thinking})
            for ev in events(cancel):
                q_put(ev)
        except Exception as exc:
            q_put({"type": "error", "message": repr(exc)})
            conv.session = None  # force a clean re-prefill next turn
            conv.open_reply = None
        finally:
            chat["busy"] = False
            q_put(None)

    def q_put(ev):
        loop.call_soon_threadsafe(q.put_nowait, ev)

    mlx_thread.submit(work)

    async def sse():
        try:
            while (ev := await q.get()) is not None:
                yield f"data: {json.dumps(ev)}\n\n"
        finally:  # client went away: stop generating
            cancel.set()

    return StreamingResponse(sse(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# ------------------------------------------------------------------ coding agent
class CodeSendReq(BaseModel):
    text: str
    thinking: bool | None = None
    reasoning_effort: str | None = None
    temperature: float | None = None


def box() -> Sandbox:
    return code["box"]


@app.get("/api/code/status")
def code_status():
    b = box()
    docker = b.docker_ok()
    running = docker and b.running()
    a: CodeAgent | None = code["agent"]
    return {
        "docker": docker,
        "image": docker and b.image_ok(),
        "running": running,
        "starting": code["starting"],
        "preview": running and b.preview_up(),
        "preview_url": f"http://localhost:{b.port}/",
        "busy": code["busy"],
        "context": a.context if a else 0,
        "max_context": a.max_context if a else min(MAX_CONTEXT, max(1024, getattr(state.get("engine"), "context_limit", MAX_CONTEXT + 2048) - 2048)),
        "plan": a.plan if a else "",
        "limits": b.limits,
    }


@app.post("/api/code/start")
async def code_start():
    if code["starting"]:
        raise HTTPException(409, "the sandbox is already starting")

    def target(put):
        code["starting"] = True
        try:
            box().ensure(lambda line: put({"type": "log", "line": line}))
            put({"type": "ready"})
        finally:
            code["starting"] = False

    return sse_response(target)


@app.post("/api/code/stop")
def code_stop():
    code["cancel"].set()
    box().kill_all(include_autostart=False)  # the agent's commands; the dev server keeps running
    return {"ok": True}


@app.post("/api/code/reset")
def code_reset():
    code["cancel"].set()
    box().kill_all()
    for _ in range(200):  # let the agent loop notice the cancel
        if not code["busy"]:
            break
        time.sleep(0.05)
    if code["busy"]:
        raise HTTPException(409, "the agent did not stop in time; try again")
    code["agent"] = None
    code["events"] = []  # new log; subscribers see the epoch change and start over
    code["epoch"] += 1
    try:
        box().reset()
    except SandboxError as exc:
        raise HTTPException(500, str(exc))
    return {"ok": True}


@app.post("/api/code/send")
async def code_send(req: CodeSendReq):
    e = engine()
    if not await asyncio.to_thread(box().running):
        raise HTTPException(409, "the sandbox is not running; start it first")
    # No await from here until busy is set, so concurrent sends can't both pass the check.
    if code["busy"]:
        # Steering (Codex / Claude Code): queue it; the agent sees it at the next step boundary.
        code["agent"].steer(req.text)
        code_emit({"type": "user", "text": req.text, "queued": True})
        return {"ok": True, "queued": True}
    if msg := busy(except_="code"):
        raise HTTPException(409, msg)
    if code["agent"] is None:
        code["agent"] = CodeAgent(e, box())
    agent: CodeAgent = code["agent"]
    for k in ("thinking", "reasoning_effort", "temperature"):
        if getattr(req, k) is not None:
            setattr(agent, k, getattr(req, k))
    cancel = code["cancel"]
    cancel.clear()
    code["busy"] = True
    code_emit({"type": "user", "text": req.text})

    def work():  # the run is independent of any client connection; Stop cancels it
        text = req.text
        try:
            while True:
                agent.run(text, code_emit, cancel, on_mlx=lambda fn: run(fn))
                # Messages queued after the agent's last check would otherwise be lost: run them now.
                if cancel.is_set() or not agent.steering:
                    break
                text = agent.steering.popleft()
        except Exception as exc:
            agent.reset_context()  # the transcript may be inconsistent: start a fresh context next time
            code_emit({"type": "error", "message": str(exc) if isinstance(exc, SandboxError) else repr(exc)})
            code_emit({"type": "done", "stop": "error", "steps": 0, "tool_calls": 0, "context": 0})
        finally:
            code["busy"] = False

    threading.Thread(target=work, daemon=True, name="code-agent").start()
    return {"ok": True}


@app.get("/api/code/events")
async def code_events(since: int = 0, epoch: int = -1):
    """Replay the event log from `since`, then follow it live. Reconnect-safe: a client that
    knows another epoch (session before a reset, or none) first gets a `reset` event."""

    async def stream():
        known, i, last = epoch, since, time.monotonic()
        while True:
            if code["epoch"] != known:  # client is on another session: start it over
                known, i = code["epoch"], 0
                yield f"data: {json.dumps({'type': 'reset', 'epoch': known, 'seq': -1})}\n\n"
            log = code["events"]
            while i < len(log):
                yield f"data: {json.dumps(log[i])}\n\n"
                i += 1
                last = time.monotonic()
            if time.monotonic() - last > 15:
                yield ": ping\n\n"
                last = time.monotonic()
            await asyncio.sleep(0.05)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/code/transcript")
def code_transcript():
    """The agent's whole context exactly as the model sees it (system prompt, tools, calls, results)."""
    a: CodeAgent | None = code["agent"]
    if a is None or a.session is None:
        return {"text": "", "tokens": 0}
    tokens = list(a.session.tokens)
    return {"text": a.e.decode(tokens), "tokens": len(tokens)}


@app.get("/api/code/image")
def code_image(path: str):
    """An image a hook produced inside the sandbox (e.g. a page screenshot); only under /tmp/bonsai/."""
    if not path.startswith("/tmp/bonsai/") or ".." in path or not path.endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(400, "not a hook image path")
    try:
        data = box().read(path)
    except SandboxError as exc:
        raise HTTPException(404, str(exc))
    return Response(data, media_type="image/jpeg" if path.endswith(("jpg", "jpeg")) else "image/png",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/code/diff")
def code_diff(a: str, b: str = "HEAD"):
    try:
        return {"diff": box().diff(a, b)}
    except SandboxError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/code/checkpoints")
def code_checkpoints():
    try:
        return {"checkpoints": box().checkpoints()}
    except SandboxError as exc:
        raise HTTPException(409, str(exc))


class RestoreReq(BaseModel):
    id: str


@app.post("/api/code/restore")
def code_restore(req: RestoreReq):
    """Put the workspace back to a checkpoint (files only; the agent is told on its next message)."""
    if code["busy"]:
        raise HTTPException(409, "stop the agent before restoring a checkpoint")
    try:
        box().restore(req.id)
    except SandboxError as exc:
        raise HTTPException(400, str(exc))
    if (a := code["agent"]) is not None:
        a.note_restore(req.id)
    code_emit({"type": "files", "tree": box().tree()})
    code_emit({"type": "note", "text": f"Workspace restored to checkpoint {req.id}."})
    return {"ok": True}


@app.get("/api/code/tree")
def code_tree():
    try:
        return {"files": box().tree()}
    except SandboxError as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/code/file")
def code_file(path: str):
    try:
        data = box().read(path)
    except SandboxError as exc:
        raise HTTPException(404, str(exc))
    if b"\0" in data[:4096]:
        return {"path": path, "binary": True, "size": len(data), "text": ""}
    return {"path": path, "binary": False, "size": len(data), "text": data[:1_000_000].decode(errors="replace")}


@app.get("/api/code/archive")
def code_archive():
    try:
        data = box().archive()
    except SandboxError as exc:
        raise HTTPException(409, str(exc))
    return Response(data, media_type="application/gzip",
                    headers={"Content-Disposition": 'attachment; filename="workspace.tar.gz"'})


if DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/")
    def index():
        return FileResponse(DIST / "index.html")


def _shutdown():
    """Stop everything and really exit. Without this, Ctrl+C left an orphaned process: uvicorn waited
    forever for the never-ending /api/code/events stream, and the MLX worker thread (non-daemon) kept
    the interpreter alive while `uv run` had already returned to the prompt."""
    print("Shutting down: stopping the agent and removing the sandbox…", flush=True)
    chat["cancel"].set()
    code["cancel"].set()
    try:
        box().remove()
    except Exception as exc:  # report, then exit anyway
        print(f"could not remove the sandbox container: {exc}", flush=True)
    e = state.get("engine")
    if e is not None and hasattr(e, "close"):
        e.close()
    shutil.rmtree(UPLOADS, ignore_errors=True)
    os._exit(0)


def _port_taken(host: str, port: int) -> bool:
    """True if a server is accepting connections there (a bind test also fails for sockets that are only in
    TIME_WAIT after a restart, which uvicorn can reuse)."""
    import socket

    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


if __name__ == "__main__":
    # Another server already owns the port and the sandbox: leave both alone (cleaning up here removed the
    # running server's sandbox container, workspace included).
    if _port_taken("127.0.0.1", 8000):
        print("Another Bonsai server is already running on 127.0.0.1:8000; not starting a second one "
              "(stop it with Ctrl+C in its terminal first).", flush=True)
        sys.exit(1)
    # The sandbox lives only as long as the server (also cleaned at start, in case of a hard kill).
    threading.Thread(target=lambda: box().remove(), daemon=True).start()
    mlx_thread.submit(_load)
    try:
        # Open streams (SSE) get 2 s to close on shutdown instead of being waited on forever.
        uvicorn.run(app, host="127.0.0.1", port=8000, timeout_graceful_shutdown=2)
    finally:
        _shutdown()
