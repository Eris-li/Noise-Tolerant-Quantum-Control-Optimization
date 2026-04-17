FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /workspace

RUN mkdir -p /tmp/matplotlib

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
COPY experiments ./experiments
COPY scripts ./scripts
COPY docs ./docs
COPY artifacts ./artifacts

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip install --no-cache-dir -e .

CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]
