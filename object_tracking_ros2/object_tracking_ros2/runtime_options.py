"""Shared full-system launch defaults and validation (no hardware side effects)."""
import math

# name: (default, help)
OPTIONS = {
    'show_rrt': ('true', 'Display live RRT* search and the checked route in RViz'),
    'approach_camera_fov_fallback': ('true', 'Try camera-FOV orientation only after strict approach search fails'),
    'controller_language': ('cpp', 'cpp rclcpp frontend or python rclpy frontend'),
    'tracking_mode': ('approaching', 'keep_distance or approaching'),
    'cone_distance_mm': ('350.0', 'Fixed distance, or initial approach distance, in mm'),
    'cone_half_angle_deg': ('30.0', 'Cone half-angle in degrees'),
    'approach_final_distance_mm': ('150.0', 'Final approach cone distance in mm'),
    'approach_step_mm': ('20.0', 'Distance reduction per accepted stage in mm'),
    'approach_distance_tolerance_mm': ('10.0', 'Axial approach allowance, plus/minus mm'),
    'target_deadband_mm': ('30.0', 'Target jitter sphere radius in robot-base mm'),
    'sdk_speed': ('70', 'SDK speed setting, 1 through the configured ceiling of 80'),
    'reference_speed_rad_s': ('1.0', 'Requested joint-vector speed before SDK-dependent cap'),
    'camera_backend': ('orbbec', 'Detector camera backend'),
    'camera_rate': ('30.0', 'Requested detector camera frame rate'),
}


def validate(options):
    values = dict(options)
    for key in ('show_rrt', 'approach_camera_fov_fallback'):
        value = values.get(key, 'true')
        if not isinstance(value, bool):
            if value not in ('true', 'false'):
                raise ValueError(f'{key} must be true or false')
            value = value == 'true'
        values[key] = value
    if values.get('controller_language', 'cpp') not in ('cpp', 'python'):
        raise ValueError('controller_language must be cpp or python')
    if values['tracking_mode'] not in ('keep_distance', 'approaching'):
        raise ValueError('tracking_mode must be keep_distance or approaching')
    if values['camera_backend'] not in ('orbbec','realsense','auto'):
        raise ValueError('camera_backend must be orbbec, realsense or auto')
    limits = {'cone_distance_mm': (10., 500.), 'cone_half_angle_deg': (1., 45.),
              'approach_final_distance_mm': (10., 500.), 'approach_step_mm': (1., 50.),
              'approach_distance_tolerance_mm': (.001, 25.), 'target_deadband_mm': (0., 500.),
              'reference_speed_rad_s': (.001, 1.), 'camera_rate': (1., 60.)}
    for key, (low, high) in limits.items():
        number = float(values[key])
        if not math.isfinite(number) or not low <= number <= high:
            raise ValueError(f'{key} must be between {low} and {high}')
        values[key] = number
    speed = float(values['sdk_speed'])
    if not math.isfinite(speed) or not speed.is_integer() or not 1 <= speed <= 80:
        raise ValueError('sdk_speed must be an integer from 1 to 80')
    values['sdk_speed'] = int(speed)
    if values['tracking_mode'] == 'approaching':
        if values['approach_final_distance_mm'] > values['cone_distance_mm']:
            raise ValueError('Approach final distance cannot exceed the initial cone distance')
        if values['approach_distance_tolerance_mm'] > min(values['approach_step_mm']/2,
                                                          values['approach_final_distance_mm']-1):
            raise ValueError('Approach allowance must be at most half the stage step and below the final distance')
    return values


def controller_overrides(values):
    keys = ('cone_distance_mm', 'cone_half_angle_deg', 'approach_final_distance_mm',
            'approach_step_mm', 'approach_distance_tolerance_mm', 'target_deadband_mm',
            'sdk_speed', 'reference_speed_rad_s')
    return {**{key: values[key] for key in keys}, 'show_rrt': values['show_rrt'],
            'approach_camera_fov_fallback': values['approach_camera_fov_fallback'],
            'approach_enabled': values['tracking_mode'] == 'approaching'}
