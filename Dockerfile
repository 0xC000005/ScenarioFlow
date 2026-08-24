# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.9.2-python3.13-bookworm-slim AS source
WORKDIR /build
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src

FROM source AS python-builder
RUN uv venv .venv \
    && uv export --quiet --frozen --no-dev --no-emit-project --prune torch \
        --extra demo-server --extra demo-openai \
        --output-file /tmp/requirements.txt \
    && uv pip install --python .venv/bin/python --requirement /tmp/requirements.txt \
    && uv pip install --python .venv/bin/python torch==2.13.0 --torch-backend cpu \
    && uv pip install --python .venv/bin/python --no-deps .

FROM node:22.18.0-bookworm-slim AS codex-builder
RUN npm install --global @openai/codex@0.149.0

FROM node:22.18.0-bookworm-slim AS web-builder
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web ./
RUN npm run build

FROM python:3.13.9-slim-bookworm AS runtime
RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=python-builder /build/.venv /build/.venv
COPY --from=codex-builder /usr/local/bin/node /usr/local/bin/node
COPY --from=codex-builder /usr/local/lib/node_modules/@openai/codex \
    /usr/local/lib/node_modules/@openai/codex
COPY artifacts/manifest-v1.tsv /app/artifacts/manifest-v1.tsv
RUN ln -s ../lib/node_modules/@openai/codex/bin/codex.js /usr/local/bin/codex \
    && mkdir -p /app/data /app/artifacts/cache /app/results /run/scenarioflow/codex
ENV PATH="/build/.venv/bin:/usr/local/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    SCENARIOFLOW_ARTIFACT_MANIFEST=/app/artifacts/manifest-v1.tsv \
    CODEX_HOME=/run/scenarioflow/codex
CMD ["scenarioflow", "--help"]

FROM nginx:1.29-alpine AS web
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=web-builder /web/dist /usr/share/nginx/html
