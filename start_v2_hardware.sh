#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MYCOBOT_IK_SEED="${MYCOBOT_IK_SEED:-1}"
.venv/bin/python -c 'from webui.hardware_diffusion import verified_status; s=verified_status(); print("Pinned v2 ready:", s["checkpoint_sha256"]); print("Starting hardware UI; startup does not connect or move the arm.")'
exec ./start_webui.sh --hardware
