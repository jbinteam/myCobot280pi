#!/usr/bin/env bash
# One terminal: detector, direct ROS arm controller, visualization and RViz.
set -eo pipefail
ros_arm_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/jazzy/setup.bash
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    exec /usr/bin/python3 "$ros_arm_root/scripts/ros_system.py" launch --help
fi
colcon --log-base "$ros_arm_root/output/object_follow_ros2/log" build \
    --base-paths "$ros_arm_root/object_tracking_ros2" "$ros_arm_root/object_tracking_cpp" \
    --build-base "$ros_arm_root/output/object_follow_ros2/build" \
    --install-base "$ros_arm_root/output/object_follow_ros2/install" \
    --packages-select object_tracking_ros2 object_tracking_cpp --cmake-args -DCMAKE_BUILD_TYPE=Release
source "$ros_arm_root/output/object_follow_ros2/install/setup.bash"
# Accept native ROS launch arguments as well as the existing --flag syntax.
for ros_arm_arg in "$@"; do
    if [[ "$ros_arm_arg" == *:=* ]]; then
        exec ros2 launch object_tracking_ros2 full_system.launch.py "$@"
    fi
done
exec /usr/bin/python3 "$ros_arm_root/scripts/ros_system.py" launch "$@"
