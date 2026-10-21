"""Read-only RViz meshes, accepted target/cone and tracking state."""
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from functools import lru_cache
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import Point, PointStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from scipy.spatial.transform import Rotation
from simulation.kinematics import ARM, URDF, origin_matrix


class TrackingVisualization(Node):
    def __init__(self, **kwargs):
        super().__init__('arm_tracking_visualization', **kwargs)
        self.declare_parameter('show_rrt', True)
        self.declare_parameter('show_camera_fov', True)
        self.planning = {}
        self.publisher=self.create_publisher(MarkerArray,'/arm/visualization',10)
        self.planning_publisher=self.create_publisher(MarkerArray,'/arm/planning_visualization',10)
        self.q=None;self.target=None;self.state={};self.visuals=[]
        model_root=URDF.parents[2]
        for link in ET.parse(URDF).getroot().findall('link'):
            for visual in link.findall('visual'):
                mesh=visual.find('geometry/mesh')
                if mesh is None:continue
                name=mesh.get('filename')
                prefix='package://mycobot_description/'
                if not name.startswith(prefix):continue
                file=model_root/name[len(prefix):]
                color=visual.find('material/color')
                rgba=list(map(float,color.get('rgba').split())) if color is not None else [.75,.8,.9,1.]
                self.visuals.append((link.get('name'),origin_matrix(visual.find('origin')),
                    file.as_uri(),list(map(float,mesh.get('scale','1 1 1').split())),rgba))
        self.create_subscription(JointState,'/joint_states',self.joints,10)
        self.create_subscription(PointStamped,'/object_follow/target',self.point,10)
        self.create_subscription(String,'/object_follow/status',self.status,10)
        self.create_subscription(String,'/object_follow/planning_visualization',self.planning_status,10)
        self.create_timer(.1,self.publish)

    def joints(self,message):
        if len(message.name)!=len(message.position) or not set(ARM.names).issubset(message.name):return
        q=np.array([message.position[message.name.index(name)] for name in ARM.names])
        if np.isfinite(q).all():self.q=q

    def point(self,message):
        if message.header.frame_id!=ARM.base_frame:return
        p=message.point;target=np.array([p.x,p.y,p.z])
        if np.isfinite(target).all():self.target=target

    def status(self,message):
        try:self.state=json.loads(message.data)
        except ValueError:pass

    def planning_status(self, message):
        try:
            value = json.loads(message.data)
            if not isinstance(value, dict) or value.get('schema_version') != 1 or value.get('frame_id') != ARM.base_frame:
                return
            nodes, route = value.get('nodes', []), value.get('route_deg', [])
            if len(nodes) > 4096 or len(route) > 4096:
                return
            joints = [node['q_deg'] for node in nodes] + route
            if any(len(q) != 6 or not np.isfinite(q).all() for q in joints):
                return
            self.planning = value
        except (ValueError, TypeError, KeyError):
            pass

    @staticmethod
    @lru_cache(maxsize=8192)
    def tool_point(joints):
        return ARM.fk(np.deg2rad(joints)).tolist()

    def planning_markers(self):
        value = self.planning
        visible = self.get_parameter('show_rrt').value and value.get('enabled', False)
        # Explicit DELETE removes the old tree/route immediately on a toggle
        # or cancellation without removing measured-arm/target markers.
        markers = []
        for namespace, kind, color, width in (
                ('rrt_search', Marker.LINE_LIST, (1., 64/255, 207/255, 1.), .004),
                ('rrt_route', Marker.LINE_STRIP, (64/255, 128/255, 1., 1.), .006),
                ('rrt_status', Marker.TEXT_VIEW_FACING, (1., 1., 1., 1.), .018)):
            m = self.marker(namespace, 0, kind, color)
            m.lifetime = Duration(seconds=.35).to_msg()
            if not visible:
                m.action = Marker.DELETE
            elif namespace == 'rrt_search':
                nodes = value.get('nodes', [])
                points = []
                for node in nodes:
                    parent = node.get('parent', -1)
                    if isinstance(parent, int) and 0 <= parent < len(nodes):
                        points.extend((self.tool_point(tuple(nodes[parent]['q_deg'])), self.tool_point(tuple(node['q_deg']))))
                m.scale.x = width
                m.points = self.points(points)
            elif namespace == 'rrt_route':
                m.scale.x = width
                m.points = self.points([self.tool_point(tuple(q)) for q in value.get('route_deg', [])])
                if len(m.points) < 2:
                    m.action = Marker.DELETE
            else:
                m.scale.z = width
                m.pose.position.z = .59
                m.text = (f"RRT* {value.get('status', 'idle')} | attempt {value.get('attempt', 0)} | "
                    f"{len(value.get('nodes', []))} nodes / {value.get('rewires', 0)} rewires\n"
                    "Magenta: search   Blue: checked route")
            markers.append(m)
        return markers

    def marker(self,namespace,index,kind,color):
        m=Marker();m.header.frame_id=ARM.base_frame;m.header.stamp=self.get_clock().now().to_msg()
        m.ns=namespace;m.id=index;m.type=kind;m.action=Marker.ADD
        m.pose.orientation.w=1.
        m.color.r,m.color.g,m.color.b,m.color.a=[float(v) for v in color]
        return m

    @staticmethod
    def points(values):
        return [Point(x=float(p[0]),y=float(p[1]),z=float(p[2])) for p in values]

    def camera_markers(self):
        """Use measured joints and the controller's exact optical configuration."""
        colors = ((.25,.85,1.,.85), (.25,.85,1.,.07), (.2,1.,.4,1.), (.75,.95,1.,1.))
        kinds = (Marker.LINE_LIST, Marker.TRIANGLE_LIST, Marker.LINE_LIST, Marker.TEXT_VIEW_FACING)
        markers = [self.marker('camera_fov', i, kind, color)
                   for i, (kind, color) in enumerate(zip(kinds, colors))]
        for marker in markers:
            marker.lifetime = Duration(seconds=.35).to_msg()
            marker.action = Marker.DELETE
        config = self.state.get('visibility_camera')
        if self.q is None or not config or not self.get_parameter('show_camera_fov').value:
            return markers
        from simulation.camera_model import CameraModel
        try:
            camera = CameraModel(**config)
        except (ValueError, TypeError):
            return markers
        optical = camera.optical_transform(ARM.transform(self.q))
        origin = optical[:3, 3]
        def world(point):
            return optical[:3, :3] @ point + origin
        tx, ty = np.tan(np.deg2rad([camera.horizontal_fov_deg, camera.vertical_fov_deg])/2)
        def rectangle(depth):
            return [world(np.array([x*depth*tx,y*depth*ty,depth])) for x,y in
                    ((-1,-1),(1,-1),(1,1),(-1,1))]
        far = max(.5, 2*camera.near_m)
        near_ring, far_ring = rectangle(camera.near_m), rectangle(far)
        edges, surface, ray, label = markers
        edges.action = surface.action = label.action = Marker.ADD
        edges.scale.x = .002
        edge_points = []
        triangles = []
        for i in range(4):
            j = (i+1)%4
            edge_points.extend((near_ring[i],near_ring[j],far_ring[i],far_ring[j],near_ring[i],far_ring[i]))
            triangles.extend((near_ring[i],far_ring[i],far_ring[j],near_ring[i],far_ring[j],near_ring[j]))
        edges.points = self.points(edge_points)
        surface.points = self.points(triangles)
        surface.scale.x = surface.scale.y = surface.scale.z = 1.
        label.scale.z = .014
        label.pose.position.x,label.pose.position.y,label.pose.position.z = [float(v) for v in origin+[0,0,.045]]
        label.text = f"Wrist camera {camera.horizontal_fov_deg:g}° × {camera.vertical_fov_deg:g}°"
        label.text += '\nMount: assumed' if camera.mount_assumed else '\nModeled image bounds'
        if self.target is not None:
            projection = camera.project(optical[:3, :3].T @ (self.target-origin))
            ray.action = Marker.ADD; ray.scale.x = .002
            ray.points = self.points([origin,self.target])
            if not projection['in_view']:
                ray.color.r,ray.color.g,ray.color.b = 1.,.35,.1
            label.text += '\nTarget in image bounds' if projection['in_view'] else '\nTarget outside image bounds'
        return markers

    def publish(self):
        markers=self.camera_markers()
        if self.q is not None:
            transforms=ARM.link_transforms(self.q)
            for index,(link,origin,resource,scale,color) in enumerate(self.visuals):
                if link not in transforms:continue
                t=transforms[link]@origin
                m=self.marker('measured_arm',index,Marker.MESH_RESOURCE,color)
                m.mesh_resource=resource;m.mesh_use_embedded_materials=True
                m.scale.x,m.scale.y,m.scale.z=scale
                m.pose.position.x,m.pose.position.y,m.pose.position.z=[float(v) for v in t[:3,3]]
                q=Rotation.from_matrix(t[:3,:3]).as_quat()
                m.pose.orientation.x,m.pose.orientation.y,m.pose.orientation.z,m.pose.orientation.w=[float(v) for v in q]
                markers.append(m)
        lost=not self.state.get('detector_tracking')
        color=(.9,.35,.1,1.) if lost else (.15,1.,.35,1.)
        label=self.marker('state',0,Marker.TEXT_VIEW_FACING,color)
        label.pose.position.z=.55;label.scale.z=.025
        motion=self.state.get('arm_motion',{})
        mode=('ARMED:WAITING' if self.state.get('paused') else 'TRACKING') if self.state.get('armed') else ('GENERAL' if motion.get('enabled') else 'DISARMED')
        approach=self.state.get('approach',{})
        if self.state.get('armed') and not self.state.get('paused') and approach.get('enabled'):
            mode=('HOLDING' if approach.get('phase')=='holding' else 'APPROACHING')
            mode+=f":{approach.get('distance_mm',350.):g}mm"
        label.text=mode
        if self.state.get('gaze',{}).get('orientation_mode') == 'camera_fov':
            label.text += ' · FOV fallback'
        markers.append(label)
        if self.target is not None and self.q is not None:
            eef=ARM.fk(self.q);delta=self.target-eef;length=np.linalg.norm(delta)
            target=self.marker('target',0,Marker.SPHERE,(1., .4, .05, 1.))
            target.pose.position.x,target.pose.position.y,target.pose.position.z=[float(v) for v in self.target]
            target.scale.x=target.scale.y=target.scale.z=.025;markers.append(target)
            if length>1e-6:
                axis=delta/length
                # Prefer the planner's actual held region when available.
                region=self.state.get('gaze',{}).get('tracking_region')
                if region:
                    center=np.asarray(region['center_mm'])/1000
                    axis=np.asarray(region['axis']);apex=np.asarray(region['cursor_mm'])/1000
                    radius=region['radius_mm']/1000
                else:
                    settings=self.state.get('tracking_region',{})
                    distance=settings.get('distance_mm',350.)/1000
                    angle=settings.get('half_angle_deg',30.)
                    apex=self.target;center=apex-distance*axis;radius=distance*math.tan(math.radians(angle))
                basis=np.cross(axis,[0,0,1] if abs(axis[2])<.9 else [0,1,0]);basis/=np.linalg.norm(basis)
                other=np.cross(axis,basis)
                ring=[center+radius*(math.cos(a)*basis+math.sin(a)*other) for a in np.linspace(0,2*math.pi,65)]
                disk=self.marker('region',0,Marker.LINE_STRIP,(.1,.6,1.,.85));disk.scale.x=.003
                disk.points=self.points(ring);markers.append(disk)
                # The acceptable area is the bounded plane, not the cone
                # volume. Draw both sides so it remains visible from any view.
                surface=self.marker('region',3,Marker.TRIANGLE_LIST,(.1,.6,1.,.18))
                surface.scale.x=surface.scale.y=surface.scale.z=1.
                triangles=[]
                for a,b in zip(ring[:-1],ring[1:]):
                    triangles.extend((center,a,b,center,b,a))
                surface.points=self.points(triangles);markers.append(surface)
                # Stored axis points camera -> object; -axis is the requested
                # object -> camera plane normal. Both define the same plane.
                normal=self.marker('region',4,Marker.ARROW,(.1,.6,1.,.9))
                normal.scale.x=.004;normal.scale.y=.01;normal.scale.z=.015
                normal.points=self.points([center,center-.06*axis]);markers.append(normal)
                cone=self.marker('region',1,Marker.LINE_LIST,(.1,.6,1.,.35));cone.scale.x=.0015
                cone.points=self.points([p for end in ring[::8] for p in (apex,end)]);markers.append(cone)
                aim=self.marker('region',2,Marker.ARROW,color);aim.scale.x=.006;aim.scale.y=.015;aim.scale.z=.025
                aim.points=self.points([eef,self.target]);markers.append(aim)
        self.publisher.publish(MarkerArray(markers=markers))
        self.planning_publisher.publish(MarkerArray(markers=self.planning_markers()))


def main():
    rclpy.init();node=TrackingVisualization()
    try:rclpy.spin(node)
    except (KeyboardInterrupt,ExternalShutdownException):pass
    finally:node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
