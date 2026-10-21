#!/usr/bin/env bash
# Start standalone ROS arm control; arming remains a separate ROS service call.
set -eo pipefail
# Match the evaluated planner runtime without competing BLAS worker pools.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MYCOBOT_IK_SEED=1
tracking_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tracking_dry_run=true
tracking_verified=false
tracking_auto_connect=false
case "${1:-}" in
  --hardware) tracking_dry_run=false; tracking_verified=true; tracking_auto_connect=true; shift ;;
  --dry-run) shift ;;
  -h|--help)
    cat <<'HELP'
Usage: ./start_object_tracking.sh [--dry-run|--hardware] [ROS launch overrides]

Default: standalone disarmed dry run, detected box center, 30 degree cone.
Uses the compiled C++ ROS frontend; controller_language:=python selects rclpy.
Approach starts at 350 mm and reduces to 150 mm in checked 20 mm stages.
No WebUI process is needed. --hardware connects directly to the arm bridge,
using the assumed optical camera mount and verified flange/table frames.
It remains disarmed until an explicit ROS service call.
Arm now and wait for a detected object with:
  ros2 service call /object_follow/arm std_srvs/srv/Trigger '{}'
Stop with /object_follow/stop, /arm/stop, or Ctrl+C.
Connect/disconnect with /object_follow/connect and /object_follow/disconnect.
Use backend:=http auto_connect:=false to select the legacy WebUI backend.

Example calibrated extrinsic configuration:
  ./start_object_tracking.sh --hardware config:=/absolute/path/object_follow.yaml
The detector is separate: objectScanner/scripts/collect_process/07_run_live.sh.
HELP
    exit 0 ;;
esac
source /opt/ros/jazzy/setup.bash
if [[ ! -x "$tracking_root/.venv/bin/python" ]]; then
  echo 'Missing project .venv/bin/python; install the project runtime first.' >&2
  exit 1
fi
# Build every launch so recently changed Python/config files cannot leave a
# silently stale installed ROS node. The incremental package build is small.
colcon --log-base "$tracking_root/output/object_follow_ros2/log" build \
    --base-paths "$tracking_root/object_tracking_ros2" "$tracking_root/object_tracking_cpp" \
    --build-base "$tracking_root/output/object_follow_ros2/build" \
    --install-base "$tracking_root/output/object_follow_ros2/install" \
    --packages-select object_tracking_ros2 object_tracking_cpp --cmake-args -DCMAKE_BUILD_TYPE=Release
source "$tracking_root/output/object_follow_ros2/install/setup.bash"
# Keep the project planner/SDK importable while retaining ROS's Python paths;
# executing through the project venv supplies its matching Torch/SciPy runtime.
export PYTHONPATH="$tracking_root:$tracking_root/pymycobot${PYTHONPATH:+:$PYTHONPATH}"
exec ros2 launch object_tracking_ros2 object_follow.launch.py \
  "project_root:=$tracking_root" \
  "dry_run:=$tracking_dry_run" "frame_verified:=$tracking_verified" \
  "scene_verified:=$tracking_verified" "auto_connect:=$tracking_auto_connect" \
  "python_executable:=$tracking_root/.venv/bin/python" "$@"
