"""Full direct-hardware system with validated operator-facing launch arguments.

The owned-process supervisor retains camera/arm exclusivity, one-shot arming,
clean Stop, process-group cleanup, and compatibility with stop_ros_system.sh.
"""
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from object_tracking_ros2.runtime_options import OPTIONS, validate


def _project_root():
    for parent in Path(__file__).resolve().parents:
        if (parent/'scripts/ros_system.py').is_file():
            return str(parent)
    return str(Path.home()/'codex/projects/myCobot280pi')


def _start(context):
    def value(key):
        return LaunchConfiguration(key).perform(context)
    def boolean(key):
        result = value(key).lower()
        if result not in ('true', 'false'):
            raise ValueError(f'{key} must be true or false')
        return result == 'true'
    settings = validate({key: value(key) for key in OPTIONS})
    root = Path(value('project_root')).expanduser().resolve()
    if not (root/'scripts/ros_system.py').is_file():
        raise ValueError('project_root must point to the myCobot280pi checkout')
    command = ['/usr/bin/python3', str(root/'scripts/ros_system.py'), 'start']
    for key in ('scanner_root', 'dataset', 'category', 'robot_host', 'robot_port'):
        command += ['--'+key.replace('_','-'), value(key)]
    for key, setting in settings.items():
        command += ['--'+key.replace('_','-'), str(setting).lower() if isinstance(setting,bool) else str(setting)]
    for key, flag, enabled in [('auto_arm','--no-arm',False), ('rviz','--no-rviz',False),
                               ('detector','--no-detector',False), ('restart','--restart',True),
                               ('print_config','--print-config',True)]:
        if boolean(key) == enabled:
            command.append(flag)
    process = ExecuteProcess(cmd=command, cwd=str(root), output='screen',
                             sigterm_timeout='30', sigkill_timeout='35')
    return [RegisterEventHandler(OnProcessExit(target_action=process,
                on_exit=[EmitEvent(event=Shutdown(reason='Full-system supervisor exited'))])), process]


def generate_launch_description():
    root = _project_root()
    arguments = {
        'project_root': (root, 'Path to the myCobot280pi checkout'),
        'scanner_root': (str(Path(root).parent/'objectScanner'), 'Path to objectScanner checkout'),
        'dataset': ('my_object_20260925_r2', 'Pinned detector dataset/session name'),
        'category': ('my_object', 'Detector object category'),
        'robot_host': ('192.168.77.225', 'Robot Pi bridge address'),
        'robot_port': ('9000', 'Robot Pi bridge TCP port'),
        'auto_arm': ('true', 'Arm once after fresh feedback; wait idle for a target'),
        'rviz': ('true', 'Open the full-system RViz display'),
        'detector': ('true', 'Start the camera/detector; false uses existing ROS detections'),
        'restart': ('false', 'Stop a previous managed run first'),
        'print_config': ('false', 'Print resolved configuration and exit without starting hardware'),
        **OPTIONS,
    }
    return LaunchDescription([DeclareLaunchArgument(key, default_value=default, description=description)
                              for key, (default, description) in arguments.items()]+[OpaqueFunction(function=_start)])
