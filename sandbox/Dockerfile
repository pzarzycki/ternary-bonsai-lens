# Sandbox for the Code tab's agent. Layers (docs/coding-agent-architecture.md):
#   /opt/bonsai     environment: AGENTS.md (global memory), settings.json, skills/, tools/ (check scripts)
#   /opt/template   a starter project (Vite react-ts) with its own AGENTS.md and .bonsai/settings.json hooks
FROM node:22-bookworm-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv git ripgrep curl ca-certificates procps less \
    bzip2 xz-utils unzip file jq \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Starter project: React + TypeScript + Vite with Tailwind v4, lucide-react, three + react-three-fiber + drei.
RUN npm create vite@latest /opt/template -- --template react-ts --no-interactive \
 && cd /opt/template && npm install \
 && npm install tailwindcss @tailwindcss/vite lucide-react three @react-three/fiber @react-three/drei \
 && npm install -D @types/three && npm cache clean --force \
 && sed -i 's/"noUnusedLocals": true/"noUnusedLocals": false/; s/"noUnusedParameters": true/"noUnusedParameters": false/' tsconfig.app.json \
 && rm -rf src/App.css src/assets public/icons.svg && sed -i 's|<title>.*</title>|<title>App</title>|' index.html

# Headless Chromium (software WebGL via SwiftShader) for the page check.
RUN mkdir -p /opt/bonsai/tools && cd /opt/bonsai/tools && npm init -y >/dev/null && npm install playwright-core@1 \
 && npx -y playwright@1 install --with-deps chromium-headless-shell && npm cache clean --force

COPY bonsai/ /opt/bonsai/
# pyapi: API lookup (signatures, docs, members, versions) in the project's own environment.
RUN printf '#!/bin/sh\nexec uv run --project /workspace --quiet python /opt/bonsai/tools/pyapi.py "$@"\n' > /usr/local/bin/pyapi \
 && printf '#!/bin/sh\nexec node /opt/bonsai/tools/check-page.mjs "$@"\n' > /usr/local/bin/check-page \
 && chmod +x /usr/local/bin/pyapi /usr/local/bin/check-page
COPY template/ /opt/template/

# /workspace starts as a ready uv project: pyproject.toml (with the CPU PyTorch index) + an empty .venv.
COPY workspace/ /workspace/
RUN cd /workspace && uv venv -q && uv sync -q

# Environment for tools in this image (the harness itself adds only a plain-terminal env, see sandbox.py).
ENV HOST=0.0.0.0 npm_config_yes=true npm_config_fund=false npm_config_audit=false npm_config_update_notifier=false
LABEL bonsai.layout="v3"
WORKDIR /workspace
CMD ["sleep", "infinity"]
