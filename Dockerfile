FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /workspace

RUN mkdir -p /tmp/matplotlib && \
    apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
COPY experiments ./experiments
COPY scripts ./scripts
COPY docs ./docs
COPY artifacts ./artifacts
COPY rydcalc ./rydcalc
COPY patches ./patches

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip install --no-cache-dir -e '.[rydcalc]' && \
    python scripts/build_rydcalc_extension.py && \
    python -c "from neutral_yb.external.rydcalc_adapter import build_yb171_atom; atom = build_yb171_atom(use_db=False); print(atom.name)"

CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]
