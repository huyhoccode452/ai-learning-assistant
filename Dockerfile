FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so this layer is cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Non-root user; data/ is where uploads and SQLite live (bind-mounted by compose).
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data/uploads \
    && chown -R appuser:appuser /app/data

# Only the runtime code. .dockerignore is an allowlist (app/, web/ if present,
# requirements.txt), so this works whether or not web/ exists yet. .env is never copied.
COPY --chown=appuser:appuser . .

USER appuser
ENV HOST=0.0.0.0 \
    PORT=8000
EXPOSE 8000
# python -m app.main reads HOST and PORT (Render sets PORT itself; 8000 is only the default).
CMD ["python", "-m", "app.main"]
