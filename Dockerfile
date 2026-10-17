# GradeLens — AI Student Performance Analyst
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    GRADELENS_MODEL_DIR=/app/models \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Unprivileged user: the app (and the code sandbox it spawns) never runs as root.
RUN useradd --create-home --uid 10001 appuser

# Dependencies first, so code changes don't reinstall them. All are prebuilt wheels: no compiler needed.
COPY requirements.txt .
RUN pip install --only-binary=:all: -r requirements.txt

COPY --chown=appuser:appuser . .
RUN python scripts/generate_sample_data.py > /dev/null \
    && mkdir -p /app/models \
    && chown -R appuser:appuser /app/models /app/sample_data

USER appuser
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import sys, urllib.request as u; sys.exit(0 if u.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).read() == b'ok' else 1)"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
