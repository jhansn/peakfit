FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

WORKDIR /app

# Faster, more deterministic installs in containers.
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1

# Cache dependency resolution first.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY tests ./tests

# Install project + dev tools (pytest, ruff) from lockfile.
RUN uv sync --frozen --group dev


FROM base AS lint
RUN uv run ruff check .


FROM base AS build
RUN uv build


FROM base AS test
RUN uv run pytest -q tests

