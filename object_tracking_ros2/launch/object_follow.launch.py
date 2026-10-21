from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
import sys

EXTRA = {key:float for key in ('cone_distance_mm','cone_half_angle_deg',
    'approach_final_distance_mm','approach_step_mm','approach_distance_tolerance_mm',
    'target_deadband_mm','reference_speed_rad_s')}
EXTRA.update(sdk_speed=int, approach_enabled=bool, show_rrt=bool, approach_camera_fov_fallback=bool)

def project_root():
    for parent in Path(__file__).resolve().parents:
        if (parent/'scripts/ros_system.py').is_file():
            return str(parent)
    return str(Path.home()/'codex/projects/myCobot280pi')

def controller(context):
    language = LaunchConfiguration('controller_language').perform(context)
    if language not in ('cpp','python'):
        raise ValueError('controller_language must be cpp or python')
    settings = {'backend': LaunchConfiguration('backend'),
        'robot_host': LaunchConfiguration('robot_host'),
        'robot_port': ParameterValue(LaunchConfiguration('robot_port'), value_type=int),
        **{key:ParameterValue(LaunchConfiguration(key),value_type=bool)
           for key in ('auto_connect','dry_run','frame_verified','scene_verified')}}
    settings.update({key:ParameterValue(LaunchConfiguration(key),value_type=kind)
        for key,kind in EXTRA.items() if LaunchConfiguration(key).perform(context) != ''})
    if language == 'cpp':
        settings['project_root'] = LaunchConfiguration('project_root')
        return [Node(package='object_tracking_cpp', executable='object_follow_cpp', name='object_follow',
            output='screen', parameters=[LaunchConfiguration('config'),settings])]
    return [Node(package='object_tracking_ros2', executable='object_follow', name='object_follow', output='screen',
        prefix=[LaunchConfiguration('python_executable'), ' '],
        parameters=[LaunchConfiguration('config'),settings])]

def generate_launch_description():
    config = str(Path(get_package_share_directory('object_tracking_ros2'))/'config/object_follow.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=config),
        DeclareLaunchArgument('dry_run', default_value='true'),
        DeclareLaunchArgument('frame_verified', default_value='false'),
        DeclareLaunchArgument('scene_verified', default_value='false'),
        DeclareLaunchArgument('backend', default_value='direct'),
        DeclareLaunchArgument('controller_language', default_value='cpp'),
        DeclareLaunchArgument('project_root', default_value=project_root()),
        DeclareLaunchArgument('auto_connect', default_value='false'),
        DeclareLaunchArgument('robot_host', default_value='192.168.77.225'),
        DeclareLaunchArgument('robot_port', default_value='9000'),
        DeclareLaunchArgument('python_executable', default_value=sys.executable),
        *[DeclareLaunchArgument(key,default_value='',description='Override YAML '+key) for key in EXTRA],
        OpaqueFunction(function=controller)])
