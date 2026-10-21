#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
npm run build --prefix webui
export OPENBLAS_NUM_THREADS=1
exec .venv/bin/python webui/server.py "$@"
