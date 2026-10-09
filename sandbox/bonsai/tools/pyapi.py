"""pyapi: look up the API of the Python packages installed in /workspace's project (exact versions).

  pyapi tokenizers.trainers.BpeTrainer      signature + docs of a class/function/module
  pyapi torch.nn.functional --search attn   names in a module (and its submodules) containing "attn"
  pyapi xml.etree.ElementTree.iterparse     standard library too

Prints the installed version, the kind, the call signature, the docstring (trimmed) and, for modules and
classes, their public members with one-line signatures. Wrong names get "did you mean" suggestions.
"""

from __future__ import annotations

import difflib
import importlib
import importlib.metadata as md
import inspect
import pkgutil
import sys

DOC_LINES = 30
MAX_MEMBERS = 60


def version_of(top: str) -> str:
    if top in sys.stdlib_module_names:
        return f"standard library (Python {sys.version.split()[0]})"
    try:
        dists = md.packages_distributions().get(top, [top])
        return f"{dists[0]} {md.version(dists[0])}"
    except Exception:
        return "version unknown"


def sig(obj) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        doc = (inspect.getdoc(obj) or "").strip().split("\n", 1)[0]
        return f"  # {doc[:120]}" if doc else "(…)"


def kind(obj) -> str:
    if inspect.ismodule(obj):
        return "module"
    if inspect.isclass(obj):
        return "class"
    if callable(obj):
        return "function"
    return type(obj).__name__


def resolve(name: str):
    """Import the longest module prefix, then walk attributes. Returns (obj, error message)."""
    parts = name.split(".")
    mod, i = None, len(parts)
    while i > 0:
        try:
            mod = importlib.import_module(".".join(parts[:i]))
            break
        except ModuleNotFoundError as exc:
            if exc.name and not ".".join(parts[:i]).startswith(exc.name) and i == len(parts):
                return None, f"importing {name} failed: {exc}"
            i -= 1
        except Exception as exc:
            return None, f"importing {'.'.join(parts[:i])} failed: {type(exc).__name__}: {exc}"
    if mod is None:
        top = parts[0]
        names = sorted({m.name for m in pkgutil.iter_modules()} | set(sys.stdlib_module_names))
        near = difflib.get_close_matches(top, names, n=5)
        return None, (f"no module named {top!r} in this project's environment."
                      + (f" Did you mean: {', '.join(near)}?" if near else "")
                      + f" If it's a package, install it with: uv add {top}")
    obj, path = mod, ".".join(parts[:i])
    for attr in parts[i:]:
        if not hasattr(obj, attr):
            members = [m for m in dir(obj) if not m.startswith("_")]
            near = difflib.get_close_matches(attr, members, n=6, cutoff=0.5)
            sub = []
            if inspect.ismodule(obj) and hasattr(obj, "__path__"):
                sub = [m.name for m in pkgutil.iter_modules(obj.__path__)]
                near += difflib.get_close_matches(attr, sub, n=4, cutoff=0.5)
            return None, (f"{path} has no attribute {attr!r}."
                          + (f" Did you mean: {', '.join(dict.fromkeys(near))}?" if near else "")
                          + f" Run `pyapi {path}` to list its members.")
        obj, path = getattr(obj, attr), f"{path}.{attr}"
    return obj, None


def members(obj) -> list[str]:
    out = []
    for n in sorted(dir(obj)):
        if n.startswith("_"):
            continue
        try:
            m = getattr(obj, n)
        except Exception:
            continue
        if inspect.ismodule(obj) and inspect.ismodule(m):
            out.append(f"  {n}  (module)")
        elif callable(m):
            out.append(f"  {n}{sig(m)}"[:200] + ("  [class]" if inspect.isclass(m) else ""))
        else:
            out.append(f"  {n}: {type(m).__name__}")
    return out


def search(obj, query: str, prefix: str) -> list[str]:
    q, hits = query.lower(), []
    mods = [(prefix, obj)]
    if inspect.ismodule(obj) and hasattr(obj, "__path__"):
        for info in pkgutil.iter_modules(obj.__path__):
            try:
                mods.append((f"{prefix}.{info.name}", importlib.import_module(f"{prefix}.{info.name}")))
            except Exception:
                continue
    for mname, m in mods:
        for n in dir(m):
            if q in n.lower() and not n.startswith("_"):
                try:
                    hits.append(f"  {mname}.{n}{sig(getattr(m, n)) if callable(getattr(m, n)) else ''}"[:200])
                except Exception:
                    hits.append(f"  {mname}.{n}")
    return list(dict.fromkeys(hits))


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0
    name = argv[0]
    obj, err = resolve(name)
    if err:
        print(f"Error: {err}")
        return 1
    print(f"{name}  [{kind(obj)}]  from {version_of(name.split('.')[0])}")
    if "--search" in argv:
        q = argv[argv.index("--search") + 1] if len(argv) > argv.index("--search") + 1 else ""
        hits = search(obj, q, name)
        print(f"{len(hits)} names containing {q!r}:" if hits else f"no names containing {q!r}")
        print("\n".join(hits[:MAX_MEMBERS]))
        return 0
    if callable(obj) and not inspect.ismodule(obj):
        print(f"signature: {name.rsplit('.', 1)[-1]}{sig(obj)}")
    doc = inspect.getdoc(obj) or "(no docstring)"
    lines = doc.splitlines()
    print("\n".join(lines[:DOC_LINES]) + (f"\n[… {len(lines) - DOC_LINES} more doc lines]" if len(lines) > DOC_LINES else ""))
    if inspect.ismodule(obj) or inspect.isclass(obj):
        ms = members(obj)
        print(f"\npublic members ({len(ms)}):")
        print("\n".join(ms[:MAX_MEMBERS]) + (f"\n  … {len(ms) - MAX_MEMBERS} more (use --search)" if len(ms) > MAX_MEMBERS else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
