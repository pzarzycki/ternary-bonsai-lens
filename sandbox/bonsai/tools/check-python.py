"""Python checks for files the agent just wrote (a Bonsai Code hook; exit 0 = clean, exit 2 = problems).

  python3 /opt/bonsai/tools/check-python.py a.py pkg/b.py
- syntax: each file must compile
- shadowing: a top-level module named like a standard-library module (tokenize.py, random.py, …) breaks
  imports for everything run from that folder (torch -> inspect -> tokenize imports *your* file)
"""

import os
import sys

problems = []
for path in sys.argv[1:]:
    if not path.endswith(".py") or not os.path.exists(path):
        continue
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem in sys.stdlib_module_names and stem != "__init__":
        problems.append(f"{path}: the name shadows the standard-library module '{stem}', so `import {stem}` (also "
                        f"inside libraries like torch) loads this file instead. Rename it, e.g. to my_{stem}.py, and "
                        f"update its imports.")
    try:
        with open(path, encoding="utf-8") as f:
            compile(f.read(), path, "exec")
    except SyntaxError as exc:
        problems.append(f"{path}:{exc.lineno}: syntax error: {exc.msg}"
                        + (f"\n    {exc.text.rstrip()}" if exc.text else ""))
    except (UnicodeDecodeError, ValueError) as exc:
        problems.append(f"{path}: could not parse: {exc}")
print("\n".join(problems) if problems else "OK")
sys.exit(2 if problems else 0)
