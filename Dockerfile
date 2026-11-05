FROM ghcr.io/astral-sh/uv:0.10.6 AS uv

FROM python:3.12-slim

COPY --from=uv /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    WORDGEN_DATA_DIR=/run/wordgen/data \
    WORDGEN_MODEL_DIR=/run/wordgen/models/modelagem \
    WORDGEN_REPORTS_DIR=/run/wordgen/reports \
    WORDGEN_RUNTIME_DIR=/var/lib/wordgen

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY docs/modelagem ./docs/modelagem
COPY .streamlit ./.streamlit
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 10001 wordgen \
    && mkdir -p /var/lib/wordgen \
    && chown -R wordgen:wordgen /app /var/lib/wordgen

USER wordgen
EXPOSE 8501

CMD ["streamlit", "run", "src/wordgen/dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
