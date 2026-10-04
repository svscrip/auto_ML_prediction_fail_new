# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    MLFLOW_TRACKING_URI=sqlite:////app/artifacts/mlflow.db

WORKDIR /app

# libgomp1 is required by CatBoost at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Runtime dependencies are installed first for better layer caching.
# requirements.txt is the pinned, Poetry-generated runtime lock (see README).
COPY requirements.txt ./
RUN pip install --retries 5 --timeout 60 -r requirements.txt

# The runtime image ships neither the git binary nor a .git directory, so
# gitpython's repository lookup (used by MLflow model logging) is expected to
# fail. Silence its startup banner to keep container logs clean.
ENV GIT_PYTHON_REFRESH=quiet

# Unprivileged runtime user.
RUN useradd --create-home --uid 1000 appuser

# Application code and datasets.
COPY src/ ./src/
COPY keis7-main/train.csv keis7-main/test.csv ./keis7-main/

RUN mkdir -p /app/artifacts && chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["python", "-m"]
CMD ["src.train", "--smoke"]
