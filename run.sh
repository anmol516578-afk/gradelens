#!/usr/bin/env bash
# GradeLens one-command start (Linux / macOS):
#   creates a virtual environment, installs dependencies, generates sample data if needed, opens the app.
# Usage: ./run.sh            (PORT=8600 ./run.sh to use another port)
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  for candidate in python3.11 python3.12 python3.13 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 \
      && "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
      PYTHON="$candidate"
      break
    fi
  done
fi
if [ -z "$PYTHON" ]; then
  echo "GradeLens needs Python 3.11 or newer. Install it from https://www.python.org/downloads/ and run this again." >&2
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "Creating a virtual environment in .venv (using $PYTHON) ..."
  "$PYTHON" -m venv .venv
fi

# Reinstall only when requirements.txt has changed since the last successful install.
if ! cmp -s requirements.txt .venv/requirements.installed; then
  echo "Installing dependencies (the first run takes a few minutes) ..."
  .venv/bin/python -m pip install --quiet --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
  cp requirements.txt .venv/requirements.installed
fi

if [ ! -f sample_data/marks_long.csv ]; then
  echo "Generating sample data ..."
  .venv/bin/python scripts/generate_sample_data.py > /dev/null
fi

[ -f .env ] && echo "Using settings from .env"
echo "Starting GradeLens at http://localhost:${PORT:-8501}  (press Ctrl+C to stop)"
exec .venv/bin/python -m streamlit run app.py --server.port "${PORT:-8501}" "$@"
