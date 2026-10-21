#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if (($#)); then
  echo "Usage: ./start_v2.sh (select the mode in the UI)" >&2
  exit 2
fi
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MYCOBOT_IK_SEED="${MYCOBOT_IK_SEED:-1}"
.venv/bin/python -c 'from webui.hardware_diffusion import verified_status; s=verified_status(); print("Pinned v2 ready:", s["checkpoint_sha256"]); print("Starting in simulation; select Simulation + hardware in the UI when needed.")'
exec ./start_webui.sh
