import pytest
from object_tracking_ros2.runtime_options import OPTIONS, validate, controller_overrides


def settings(**overrides):
    return validate({**{k:v[0] for k,v in OPTIONS.items()},**overrides})


def test_keep_distance_and_approaching_map_to_controller_parameters():
    keep=controller_overrides(settings(tracking_mode='keep_distance',cone_distance_mm='250',cone_half_angle_deg='20'))
    assert keep['approach_enabled'] is False
    assert keep['cone_distance_mm']==250. and keep['cone_half_angle_deg']==20.
    approach=controller_overrides(settings(approach_final_distance_mm='120',sdk_speed='56'))
    assert approach['approach_enabled'] is True
    assert approach['approach_final_distance_mm']==120. and approach['sdk_speed']==56


@pytest.mark.parametrize('invalid',[
    {'tracking_mode':'other'}, {'cone_distance_mm':'nan'}, {'cone_half_angle_deg':'90'},
    {'sdk_speed':'81'}, {'sdk_speed':'70.5'}, {'approach_final_distance_mm':'400'},
    {'approach_distance_tolerance_mm':'11'}, {'camera_backend':'unknown'},
    {'target_deadband_mm':'-1'}, {'reference_speed_rad_s':'inf'}, {'controller_language':'other'}])
def test_invalid_full_system_options_fail_before_launch(invalid):
    with pytest.raises(ValueError):settings(**invalid)

def test_cpp_is_default_and_python_remains_selectable():
    assert settings()['controller_language']=='cpp'
    assert settings(controller_language='python')['controller_language']=='python'


def test_optional_rrt_display_validates_and_reaches_controller():
    assert controller_overrides(settings())['show_rrt'] is True
    assert controller_overrides(settings(show_rrt='false'))['show_rrt'] is False
    with pytest.raises(ValueError, match='show_rrt'):
        settings(show_rrt='yes')


def test_camera_fallback_default_and_strict_only_option_reach_controller():
    assert controller_overrides(settings())['approach_camera_fov_fallback'] is True
    assert controller_overrides(settings(approach_camera_fov_fallback='false'))['approach_camera_fov_fallback'] is False
    with pytest.raises(ValueError,match='approach_camera_fov_fallback'):
        settings(approach_camera_fov_fallback='yes')
