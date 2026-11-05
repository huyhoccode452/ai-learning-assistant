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
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
