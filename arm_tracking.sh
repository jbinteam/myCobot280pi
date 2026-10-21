#!/usr/bin/env bash
set -eo pipefail
ros_arm_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/jazzy/setup.bash
exec /usr/bin/python3 "$ros_arm_root/scripts/ros_system.py" arm "$@"
