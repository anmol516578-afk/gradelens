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

# Hosts such as Render, Railway, Cloud Run and Hugging Face Spaces choose the port through $PORT.
ENV PORT=8501
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os, sys, urllib.request as u; port = os.environ.get('PORT', '8501'); sys.exit(0 if u.urlopen(f'http://127.0.0.1:{port}/_stcore/health', timeout=4).read() == b'ok' else 1)"

# exec: Streamlit replaces the shell, so it receives stop signals from the platform directly.
CMD ["sh", "-c", "exec streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
