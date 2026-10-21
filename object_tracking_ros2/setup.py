from setuptools import setup
setup(name='object_tracking_ros2', version='0.1.0', packages=['object_tracking_ros2'],
    data_files=[('share/ament_index/resource_index/packages', ['resource/object_tracking_ros2']),
                ('share/object_tracking_ros2', ['package.xml']),
                ('share/object_tracking_ros2/launch', ['launch/object_follow.launch.py','launch/full_system.launch.py']),
                ('share/object_tracking_ros2/config', ['config/object_follow.yaml','config/full_system.rviz'])],
    install_requires=['setuptools'], zip_safe=True, maintainer='jb', maintainer_email='jb@example.com',
    description='Standalone checked arm control and accepted RGB-D centre following', license='MIT',
    entry_points={'console_scripts': ['object_follow = object_tracking_ros2.node:main']})
