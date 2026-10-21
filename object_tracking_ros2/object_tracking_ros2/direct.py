"""In-process checked arm backend, with no WebUI or HTTP server dependency.

The route names retain the follower's existing backend protocol; dispatch is a
normal Python call. Hardware ownership, measured feedback, collision checking,
encoder execution and session leases are shared with the optional WebUI.
"""
from contextlib import nullcontext
import copy
import threading

from hardware.controller import RobotController, number
from simulation.diffusion.service import PlannerBusy
from simulation.reference_shape import settings as shape_settings
from webui.motion_settings import MAX_SDK_SPEED
from webui.mouse_motion import StaleMousePlan


class DirectBackendError(RuntimeError):
    """Backend error with the same retry/status code as the wire protocol."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class DirectBackend:
    """Own one controller without automatically opening a robot connection.

    Inject a controller or controller_factory for tests. shutdown() always
    stops/disconnects this controller, including injected controllers.
    """

    planning_routes = frozenset(('/api/ik/diffusion/plan', '/api/ik/mouse-aim/plan'))
    velocity_shape = {'kind': 'smooth_trapezoid', 'rise_percent': 5, 'fall_percent': 10}
    timeout = 3.

    def __init__(self, controller=None, controller_factory=RobotController, *, camera=None):
        from simulation.camera_model import CameraModel
        if camera is not None and not isinstance(camera, CameraModel):
            raise ValueError('Direct backend requires a checked camera model')
        if controller is not None and camera is not None and controller.ik.mouse_session is not None:
            raise ValueError('Cannot change camera for an active tracking session')
        self.controller = controller if controller is not None else controller_factory()
        if camera is not None:
            self.controller.visibility_camera = camera
        self.mouse_aim_lock = threading.Lock()
        self.closed = threading.Event()

    def shutdown(self):
        with self.controller.lock:
            if self.closed.is_set():
                return
            self.closed.set()
        self.controller.shutdown()

    def call(self, route, body=None):
        return self._checked_call(route, body)

    def plan_with_observer(self, body, observer):
        return self._checked_call('/api/ik/mouse-aim/plan', body, observer)

    def _checked_call(self, route, body=None, observer=None):
        try:
            return self._call(route, body, observer)
        except DirectBackendError:
            raise
        except (StaleMousePlan, PlannerBusy) as exc:
            raise DirectBackendError(409, str(exc)) from exc
        except (ValueError, TypeError) as exc:
            raise DirectBackendError(400, str(exc)) from exc
        except Exception as exc:
            raise DirectBackendError(503, str(exc)) from exc

    def _production_settings(self, body, tracking=False):
        key = 'tracking_engine' if tracking else 'planner'
        if body.get(key, 'learned_rrt') != 'learned_rrt':
            raise ValueError('Standalone arm uses the learned IK + classical RRT* planner')
        shape = shape_settings(body.get('velocity_shape', self.velocity_shape))
        if shape != self.velocity_shape:
            raise ValueError('Standalone arm uses the verified 5% rise / 10% fall velocity profile')
        if not tracking and body.get('joint_command', 'encoder_drag_profile') != 'encoder_drag_profile':
            raise ValueError('Standalone arm uses the checked encoder velocity-profile executor')
        body.update({key: 'learned_rrt', 'velocity_shape': shape})
        if not tracking:
            body['joint_command'] = 'encoder_drag_profile'
        return body

    def _call(self, route, body, observer=None):
        c = self.controller
        if self.closed.is_set():
            raise DirectBackendError(503, 'Arm backend has shut down')
        if body is None:
            if route == '/api/feedback':
                # Never queue measured feedback behind serial command traffic.
                return c.feedback_snapshot()
            if route == '/api/state':
                with c.lock:
                    return copy.deepcopy(dict(c.snapshot(), max_sdk_speed=MAX_SDK_SPEED,
                        hardware_enabled=True, mode_selectable=False, control_mode='hardware',
                        backend='direct'))
            raise DirectBackendError(404, 'Unknown read endpoint')
        if not isinstance(body, dict):
            raise ValueError('Expected a command object')
        body = copy.deepcopy(body)
        # The planner captures and rechecks generation under the controller
        # lock itself, releasing it during expensive geometric computation.
        with nullcontext() if route in self.planning_routes else c.lock:
            if self.closed.is_set():
                raise DirectBackendError(503, 'Arm backend has shut down')
            if route == '/api/connect':
                result = c.connect(body.get('host'), body.get('port', 9000))
            elif route == '/api/disconnect':
                result = c.disconnect()
            elif route == '/api/stop':
                result = c.stop()
            elif route == '/api/ik/diffusion/plan':
                # Historical internal route spelling; the only accepted
                # planner here is learned_rrt, for both pose and joint goals.
                result = c.ik.prepare_diffusion(self._production_settings(body))
            elif route == '/api/ik/mouse-aim/start':
                result = c.ik.start_mouse_aim(self._production_settings(body, tracking=True))
            elif route == '/api/ik/mouse-aim/plan':
                if not self.mouse_aim_lock.acquire(blocking=False):
                    raise DirectBackendError(409, 'Tracking planner busy; retry the latest target')
                try:
                    result = c.ik.prepare_mouse_aim(body) if observer is None else c.ik.prepare_mouse_aim(body, observer=observer)
                finally:
                    self.mouse_aim_lock.release()
            elif route in ('/api/ik/mouse-aim/pause', '/api/ik/mouse-aim/stop'):
                if set(body)-{'session_id'}:
                    raise ValueError('Tracking pause/stop accepts a session ID only')
                fn = c.ik.pause_mouse_aim if route.endswith('/pause') else c.ik.stop_mouse_aim
                result = fn(body.get('session_id'))
            elif route == '/api/ik/mouse-aim/heartbeat':
                if set(body)-{'session_id', 'execution_id', 'plan_id'}:
                    raise ValueError('Tracking heartbeat accepts session, execution and acknowledged plan IDs only')
                result = c.ik.mouse_aim_heartbeat(body.get('session_id'), body.get('execution_id'), body.get('plan_id'))
            elif route == '/api/ik/execute':
                if set(body)-{'plan_id', 'speed', 'armed', 'frame_verified', 'scene_verified', 'session_id'}:
                    raise ValueError('Execute accepts a checked plan ID and confirmation settings only')
                speed = number(body.get('speed', 30), 'IK SDK speed', 1, MAX_SDK_SPEED, True)
                result = c.ik.execute(body.get('plan_id'), speed, body.get('armed'),
                    body.get('frame_verified'), body.get('scene_verified'), body.get('session_id'))
            elif route == '/api/ik/heartbeat':
                result = c.ik.heartbeat(body.get('execution_id'))
            else:
                raise DirectBackendError(404, 'Unknown hardware endpoint')
            return copy.deepcopy(dict(result, hardware_enabled=True, mode_selectable=False))
