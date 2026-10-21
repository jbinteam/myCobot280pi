#!/usr/bin/env bash
# Request six zero joint angles through the running standalone ROS controller.
set -eo pipefail
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: ./zero_arm.sh"
    echo "Stops tracking, enables checked joint motion, and requests all six joints at 0 degrees."
    echo "Requires the ROS arm system to be running. Tracking remains disabled."
    exit 0
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: ./zero_arm.sh" >&2
    exit 2
fi
source /opt/ros/jazzy/setup.bash
exec /usr/bin/python3 - <<'PY'
import json
import sys
import time

import rclpy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

rclpy.init()
node = rclpy.create_node('zero_arm_request')
status = {}
received = 0.0
enabled = False

def update(message):
    global received
    status.clear()
    status.update(json.loads(message.data))
    received = time.monotonic()

def call(name):
    client = node.create_client(Trigger, name)
    try:
        if not client.wait_for_service(timeout_sec=5):
            raise RuntimeError(f'{name} unavailable. Start ./start_ros_system.sh first.')
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=10)
        if not future.done():
            raise RuntimeError(f'{name} timed out')
        response = future.result()
        if not response.success:
            raise RuntimeError(f'{name}: {response.message}')
        print(response.message, flush=True)
    finally:
        node.destroy_client(client)

def wait_for(condition, seconds, error):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if condition():
            return
    raise RuntimeError(error)

def stopped():
    motion = status.get('arm_motion', {})
    return (received > stop_at and not status.get('armed', True)
            and not status.get('session_id') and not status.get('planning', True)
            and not motion.get('stop_pending', True)
            and not motion.get('planning', True) and not motion.get('execution_id'))

node.create_subscription(String, '/object_follow/status', update, 10)
publisher = node.create_publisher(JointState, '/arm/command_joints', 1)
try:
    call('/arm/stop')
    stop_at = time.monotonic()
    wait_for(stopped, 15, 'Arm did not finish stopping; zero pose was not sent.')
    call('/arm/enable_motion')
    enabled = True
    enabled_at = time.monotonic()
    wait_for(lambda: received > enabled_at
             and status.get('arm_motion', {}).get('state') == 'ready'
             and not status.get('armed', True), 5,
             'Controller did not become ready for joint motion.')
    wait_for(lambda: publisher.get_subscription_count() > 0, 5,
             'No joint-command subscriber; zero pose was not sent.')
    message = JointState()
    message.name = ['joint2_to_joint1', 'joint3_to_joint2', 'joint4_to_joint3',
                    'joint5_to_joint4', 'joint6_to_joint5', 'joint6output_to_joint6']
    message.position = [0.0] * 6
    message.header.stamp = node.get_clock().now().to_msg()
    sent_at = time.monotonic()
    publisher.publish(message)
    wait_for(lambda: received > sent_at and status.get('arm_motion', {}).get('state')
             in ('planning', 'running', 'completed', 'rejected', 'stopped'),
             5, 'Controller did not acknowledge the zero-pose request.')
    motion = status['arm_motion']
    if motion['state'] in ('rejected', 'stopped'):
        raise RuntimeError(motion.get('reason') or 'Zero-pose request rejected')
    print('Zero-pose request accepted for planning/execution. Tracking remains disabled.')
    print('Follow completion in RViz or /object_follow/status (arm_motion).')
except (Exception, KeyboardInterrupt) as exc:
    print(f'Zero arm: {exc or "Interrupted"}', file=sys.stderr)
    if enabled and rclpy.ok():
        try:
            call('/arm/stop')
        except Exception as stop_error:
            print(f'Stop: {stop_error}', file=sys.stderr)
    sys.exit(1)
finally:
    node.destroy_node()
    rclpy.try_shutdown()
PY
