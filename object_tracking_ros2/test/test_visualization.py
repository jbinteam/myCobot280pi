"""Read-only RViz geometry is based on measured joints and the held region."""
from pathlib import Path
import json
from unittest.mock import patch
import numpy as np
import pytest
pytest.importorskip('rclpy')
import rclpy
from rclpy.parameter import Parameter
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from object_tracking_ros2.visualization import TrackingVisualization
from simulation.kinematics import ARM
from simulation.tracking_geometry import region_for


def test_measured_meshes_and_350mm_cone_match_tracking_geometry(monkeypatch):
    monkeypatch.setenv('ROS_DOMAIN_ID','181')
    rclpy.init();node=TrackingVisualization()
    try:
        node.q=np.deg2rad([1,-40,-130,80,1,50])
        transform=ARM.transform(node.q)
        axis=transform[:3,2];center=transform[:3,3]
        node.target=center+.35*axis
        node.state={'armed':True,'detector_tracking':True}
        with patch.object(node.publisher,'publish') as publish:
            node.publish()
        markers=publish.call_args.args[0].markers
        meshes=[m for m in markers if m.type==Marker.MESH_RESOURCE]
        assert len(meshes)==8
        assert all(Path(m.mesh_resource.removeprefix('file://')).is_file() for m in meshes)
        ring=next(m for m in markers if m.ns=='region' and m.id==0)
        points=np.asarray([[p.x,p.y,p.z] for p in ring.points])
        np.testing.assert_allclose((points-center)@axis,0,atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(points-center,axis=1),.35*np.tan(np.deg2rad(30)),atol=1e-10)
        surface=next(m for m in markers if m.ns=='region' and m.id==3)
        assert surface.type==Marker.TRIANGLE_LIST and surface.color.a>0
        vertices=np.asarray([[p.x,p.y,p.z] for p in surface.points])
        np.testing.assert_allclose((vertices-center)@axis,0,atol=1e-10)
        normal=next(m for m in markers if m.ns=='region' and m.id==4)
        endpoints=np.asarray([[p.x,p.y,p.z] for p in normal.points])
        np.testing.assert_allclose(endpoints[1]-endpoints[0],-.06*axis,atol=1e-10)
        assert any(m.text=='TRACKING' for m in markers)
        node.state={'armed':True,'paused':True,'detector_tracking':False}
        with patch.object(node.publisher,'publish') as publish:node.publish()
        assert any(m.text=='ARMED:WAITING' for m in publish.call_args.args[0].markers)
        node.state={'armed':True,'detector_tracking':True,
            'approach':{'enabled':True,'phase':'approaching'},
            'tracking_region':{'distance_mm':150.,'half_angle_deg':30.}}
        with patch.object(node.publisher,'publish') as publish:node.publish()
        markers=publish.call_args.args[0].markers
        ring=next(m for m in markers if m.ns=='region' and m.id==0)
        points=np.asarray([[p.x,p.y,p.z] for p in ring.points])
        shrink_center=node.target-.15*axis
        np.testing.assert_allclose(np.linalg.norm(points-shrink_center,axis=1),.15*np.tan(np.deg2rad(30)),atol=1e-10)
        assert any(m.text.startswith('APPROACHING:') for m in markers)
        # The next active stage has a freshly captured centerline for the same
        # object. Draw its authoritative plane, not the previous cone axis.
        node.q[0]+=np.deg2rad(3.)
        new_axis=node.target-ARM.fk(node.q);new_axis/=np.linalg.norm(new_axis)
        assert np.linalg.norm(new_axis-axis) > .001
        active=region_for(node.target*1000,new_axis,{'distance_mm':320.,'half_angle_deg':30.})
        node.state.update(gaze={'tracking_region':active},
                          approach={'enabled':True,'phase':'approaching','distance_mm':320.})
        node.q[0]+=np.deg2rad(.5)  # Live motion does not rotate a held stage.
        with patch.object(node.publisher,'publish') as publish:node.publish()
        ring=next(m for m in publish.call_args.args[0].markers if m.ns=='region' and m.id==0)
        points=np.asarray([[p.x,p.y,p.z] for p in ring.points])
        stage_center=np.asarray(active['center_mm'])/1000
        np.testing.assert_allclose((points-stage_center)@new_axis,0,atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(points-stage_center,axis=1),active['radius_mm']/1000,atol=1e-10)
    finally:node.destroy_node();rclpy.shutdown()


def test_real_fk_tree_rewire_route_and_live_display_toggle(monkeypatch):
    monkeypatch.setenv('ROS_DOMAIN_ID', '181')
    rclpy.init(); node = TrackingVisualization()
    try:
        q = [1., -40., -130., -40., 1., 50.]
        other = [2., -40., -130., -40., 1., 50.]
        snapshot = {'schema_version':1, 'frame_id':'g_base', 'enabled':True,
            'status':'checked_route', 'attempt':2, 'rewires':1,
            'nodes':[{'q_deg':q, 'parent':-1}, {'q_deg':other, 'parent':0}],
            'route_deg':[q, other]}
        node.planning_status(String(data=json.dumps(snapshot)))
        markers = node.planning_markers()
        tree, route, label = markers
        assert tree.scale.x == .004 and route.scale.x == .006
        assert tree.color.r == 1. and route.color.b == 1.
        actual = np.array([[p.x,p.y,p.z] for p in tree.points])
        np.testing.assert_allclose(actual, [ARM.fk(np.deg2rad(q)), ARM.fk(np.deg2rad(other))])
        assert len(route.points) == 2 and 'checked_route' in label.text
        with patch.object(node.publisher, 'publish') as arm_publish, patch.object(node.planning_publisher, 'publish') as search_publish:
            node.publish()
        assert not any(m.ns.startswith('rrt_') for m in arm_publish.call_args.args[0].markers)
        assert {m.ns for m in search_publish.call_args.args[0].markers} == {'rrt_search','rrt_route','rrt_status'}
        node.set_parameters([Parameter('show_rrt', value=False)])
        assert all(m.action == Marker.DELETE for m in node.planning_markers())
        node.set_parameters([Parameter('show_rrt', value=True)])
        assert all(m.action == Marker.ADD for m in node.planning_markers())
        snapshot['enabled'] = False
        node.planning_status(String(data=json.dumps(snapshot)))
        assert all(m.action == Marker.DELETE for m in node.planning_markers())
    finally:node.destroy_node();rclpy.shutdown()


def test_camera_frustum_uses_measured_mount_and_controller_fov(monkeypatch):
    from simulation.camera_model import CameraModel
    monkeypatch.setenv('ROS_DOMAIN_ID','181')
    rclpy.init(); node=TrackingVisualization()
    try:
        node.q=np.deg2rad([1,-40,-130,-40,1,50])
        camera=CameraModel(horizontal_fov_deg=72.,vertical_fov_deg=48.,
            eef_to_optical_translation_m=(.02,0.,.01),
            eef_to_optical_quaternion_xyzw=(0.,0.,np.sin(.2),np.cos(.2)))
        node.state={'visibility_camera':camera.configuration()}
        optical=camera.optical_transform(ARM.transform(node.q))
        node.target=optical[:3,3]+.2*optical[:3,2]
        edges,surface,ray,label=node.camera_markers()
        assert all(m.action==Marker.ADD for m in (edges,surface,ray,label))
        assert '72° × 48°' in label.text and 'in image bounds' in label.text
        points=np.array([[p.x,p.y,p.z] for p in edges.points])
        local=(points-optical[:3,3])@optical[:3,:3]
        np.testing.assert_allclose(abs(local[:,0]/local[:,2]),np.tan(np.deg2rad(36)),atol=1e-10)
        np.testing.assert_allclose(abs(local[:,1]/local[:,2]),np.tan(np.deg2rad(24)),atol=1e-10)
        assert len(surface.points)==24
        node.target=optical[:3,3]-.2*optical[:3,2]
        assert 'outside image bounds' in node.camera_markers()[-1].text
        node.set_parameters([Parameter('show_camera_fov',value=False)])
        assert all(m.action==Marker.DELETE for m in node.camera_markers())
        node.set_parameters([Parameter('show_camera_fov',value=True)])
        node.state={}
        assert all(m.action==Marker.DELETE for m in node.camera_markers())
    finally:node.destroy_node();rclpy.shutdown()
