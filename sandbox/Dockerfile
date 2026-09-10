# Minimal image for the Loom sandbox runner: no network at runtime, non-root, read-only rootfs.
FROM python:3.12-slim

RUN pip install --no-cache-dir "duckdb>=1.0" "pandas>=2.2" "pyarrow>=16" "pydantic>=2.7"

# Ship only the two sandbox modules as a tiny `loom.sandbox` package.
WORKDIR /app
RUN mkdir -p /app/loom/sandbox /data \
    && touch /app/loom/__init__.py \
    && printf '"""Sandbox package (container copy)."""\n' > /app/loom/sandbox/__init__.py
COPY src/loom/sandbox/base.py /app/loom/sandbox/base.py
COPY src/loom/sandbox/runner.py /app/loom/sandbox/runner.py

RUN useradd --create-home --uid 10001 sandbox && chown -R sandbox:sandbox /app
USER sandbox
ENV PYTHONPATH=/app PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# The executor pipes a SandboxRequest JSON on stdin and reads SandboxResult on stdout.
ENTRYPOINT []
CMD ["python", "-m", "loom.sandbox.runner"]
