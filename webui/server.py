#!/usr/bin/env python3
"""Local HTTP bridge between the web UI and the myCobot TCP server."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import secrets
import socket
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pymycobot.common import ProtocolCode
from webui.motion_settings import MAX_SDK_SPEED
from simulation import service as simulation
from simulation.workspace import estimate as estimate_workspace
from simulation.diffusion import service as diffusion
from simulation.diffusion import classical_service
from webui.mouse_motion import StaleMousePlan

# Public compatibility imports for existing WebUI clients and tests.
from hardware.controller import RobotController, LIMITS, number


class Handler(SimpleHTTPRequestHandler):
    controller = None
    token = None
    hardware_enabled = False
    mode_selectable = True
    mouse_aim_lock = threading.Lock()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent / 'dist'), **kwargs)

    def log_message(self, fmt, *args):
        if '/api/state' not in self.path: super().log_message(fmt, *args)

    def reply(self, status, data):
        payload = json.dumps(data, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        try: self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError): pass  # A cancelled planner response has no recipient.

    def local_request(self):
        hostname = urlsplit('http://' + self.headers.get('Host', '')).hostname
        origin = self.headers.get('Origin')
        if hostname not in ('127.0.0.1', 'localhost'): return False
        return not origin or origin in ('http://127.0.0.1:5173', 'http://localhost:5173',
                                         'http://127.0.0.1:8000', 'http://localhost:8000')

    def control_state(self):
        with self.controller.lock:
            return dict(self.controller.snapshot(), token=self.token, max_sdk_speed=MAX_SDK_SPEED,
                        hardware_enabled=self.hardware_enabled, mode_selectable=self.mode_selectable,
                        control_mode='hardware' if self.hardware_enabled else 'simulation')

    def set_control_mode(self, body):
        if set(body) != {'mode'} or body['mode'] not in ('simulation', 'hardware'):
            raise ValueError('Choose simulation or hardware mode')
        enabled = body['mode'] == 'hardware'
        with self.controller.lock:
            if enabled and not self.mode_selectable:
                raise ValueError('This server is locked to simulation only; start ./start_v2.sh to select hardware mode')
            if enabled != self.hardware_enabled:
                # Invalidate plans and transport ownership before changing the
                # shared API gate. Selecting hardware never connects the arm.
                if self.controller.robot and self.controller.moving and not self.controller.motion:
                    self.controller.stop()
                self.controller.disconnect()
                type(self).hardware_enabled = enabled
            return self.control_state()

    def do_GET(self):
        if not self.local_request(): return self.reply(403, {'error': 'Local requests only'})
        if self.path == '/api/state':
            return self.reply(200, self.control_state())
        if self.path == '/api/feedback':
            return self.reply(200, self.controller.feedback_snapshot())
        if self.path == '/api/sim/info':
            try: return self.reply(200, simulation.info())
            except Exception as exc: return self.reply(503, {'error': str(exc)})
        if self.path.startswith('/api/'): return self.reply(404, {'error': 'Unknown endpoint'})
        super().do_GET()

    def do_POST(self):
        if not self.local_request() or not secrets.compare_digest(self.headers.get('X-Robot-Token', ''), self.token):
            return self.reply(403, {'error': 'Invalid local control session; reload the page'})
        if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
            return self.reply(415, {'error': 'Expected JSON'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 4096: raise ValueError('Invalid request size')
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict): raise ValueError('Expected a JSON object')
            route = self.path
            if route == '/api/mode':
                return self.reply(200, self.set_control_mode(body))
            if route == '/api/sim/solve':
                return self.reply(200, simulation.run(body))
            if route in ('/api/sim/diffusion', '/api/sim/classical'):
                with self.controller.lock:
                    if self.hardware_enabled or self.controller.robot or self.controller.motion or self.controller.moving:
                        return self.reply(403, {'error': 'Diffusion previews require hardware-disabled, disconnected simulation mode'})
                return self.reply(200, (classical_service if route.endswith('/classical') else diffusion).plan(body))
            if route in ('/api/sim/mouse-aim', '/api/sim/approach-plan'):
                # No live-joint capture, transport reads, hardware plan, or
                # execution worker is involved in mouse tracking.
                with self.controller.lock:
                    if self.hardware_enabled or self.controller.robot or self.controller.motion or self.controller.moving:
                        return self.reply(403, {'error': 'Mouse aim requires hardware-disabled, disconnected simulation mode'})
                if not self.mouse_aim_lock.acquire(blocking=False):
                    return self.reply(409, {'error': 'Mouse aim planner busy; retry the latest cursor'})
                try:
                    if route == '/api/sim/approach-plan':
                        if body.get('tracking_engine')!='learned_rrt' or not body.get('tracking_region'):
                            raise ValueError('Approach search display requires learned RRT* region planning')
                        body.setdefault('orientation_mode','strict_then_fov')
                        from webui.planning_stream import PlanningStream
                        self.send_response(200)
                        self.send_header('Content-Type','application/x-ndjson')
                        self.send_header('Cache-Control','no-store')
                        self.send_header('Connection','close')
                        self.end_headers()
                        self.close_connection = True
                        def write(payload):
                            self.wfile.write(payload)
                            self.wfile.flush()
                        stream = PlanningStream(write)
                        try:
                            stream.result(simulation.mouse_aim_plan({**body,'_rrt_observer':stream.observe}))
                        except Exception as exc:
                            stream.error(exc)
                        return
                    return self.reply(200, simulation.mouse_aim_plan(body))
                finally:
                    self.mouse_aim_lock.release()
            if route == '/api/sim/prm':
                # A dry-run query can use captured live joints, but never creates
                # a HardwareIK plan or invokes its execution worker.
                with self.controller.lock:
                    if self.controller.robot:
                        self.controller._read()
                        if self.controller.motion or self.controller.moving:
                            raise ValueError('Wait for the robot to stop before planning')
                        body = {**body, 'start_deg': self.controller.angles[:]}
                        start_source = 'live robot joints'
                    else:
                        start_source = 'simulation starting pose'
                return self.reply(200, {**simulation.plan(body), 'start_source': start_source})
            if route == '/api/sim/workspace':
                with self.controller.lock:
                    if self.controller.robot:
                        self.controller._read()
                        if self.controller.motion or self.controller.moving:
                            raise ValueError('Wait for the robot to stop before estimating solvable targets')
                        body = {**body, 'start_deg':self.controller.angles[:]}
                        start_source = 'live robot joints'
                    else:
                        start_source = 'simulation starting pose'
                return self.reply(200, {**estimate_workspace(body), 'start_source':start_source})
            # Planning already captures/rechecks the controller generation and
            # releases its lock for expensive computation. Short commands hold
            # the same lock as mode changes so a disabled gate cannot race a send.
            planning_routes = ('/api/ik/plan', '/api/ik/diffusion/plan', '/api/ik/joint-diagnostic/plan', '/api/ik/jog-qualification/plan', '/api/ik/mouse-aim/plan', '/api/ik/mouse-aim/plan-stream')
            with nullcontext() if route in planning_routes else self.controller.lock:
                if not self.hardware_enabled and (route.startswith('/api/ik/') or route in ('/api/connect', '/api/move', '/api/jog')):
                    return self.reply(403, {'error': 'Simulation only; hardware commands are disabled'})
                if route == '/api/connect': result = self.controller.connect(body.get('host'), body.get('port', 9000))
                elif route == '/api/disconnect': result = self.controller.disconnect()
                elif route == '/api/move': result = self.controller.move(body.get('angles'), body.get('speed',30), body.get('armed'))
                elif route == '/api/jog': result = self.controller.jog(body.get('joint'),body.get('direction'),body.get('speed',30),body.get('duration',0.5),body.get('armed'))
                elif route == '/api/ik/plan': result = self.controller.ik.prepare(body)
                elif route == '/api/ik/diffusion/plan': result = self.controller.ik.prepare_diffusion(body)
                elif route == '/api/ik/encoder-drag/preflight': result = self.controller.ik.encoder_preflight(body)
                elif route == '/api/ik/small-target': result = self.controller.ik.small_target(body)
                elif route == '/api/ik/joint-diagnostic/plan': result = self.controller.ik.prepare_joint_diagnostic(body)
                elif route == '/api/ik/jog-qualification/plan': result = self.controller.ik.prepare_jog_qualification(body)
                elif route == '/api/ik/mouse-aim/start': result = self.controller.ik.start_mouse_aim(body)
                elif route in ('/api/ik/mouse-aim/plan', '/api/ik/mouse-aim/plan-stream'):
                    if not self.mouse_aim_lock.acquire(blocking=False):
                        return self.reply(409, {'error': 'Mouse aim planner busy; retry the latest cursor'})
                    try:
                        if route.endswith('plan-stream'):
                            from webui.planning_stream import PlanningStream
                            self.send_response(200)
                            self.send_header('Content-Type','application/x-ndjson')
                            self.send_header('Cache-Control','no-store')
                            self.send_header('Connection','close')
                            self.end_headers()
                            self.close_connection = True
                            def write(payload):
                                self.wfile.write(payload)
                                self.wfile.flush()
                            stream = PlanningStream(write)
                            try:
                                stream.result(self.controller.ik.prepare_mouse_aim(body, observer=stream.observe))
                            except (StaleMousePlan, diffusion.PlannerBusy) as exc:
                                stream.error(exc, 409)
                            except (ValueError, TypeError) as exc:
                                stream.error(exc, 400)
                            except Exception as exc:
                                stream.error(exc)
                            return
                        result = self.controller.ik.prepare_mouse_aim(body)
                    finally: self.mouse_aim_lock.release()
                elif route == '/api/ik/mouse-aim/stop':
                    if set(body)-{'session_id'}: raise ValueError('Mouse aim stop accepts a session ID only')
                    result = self.controller.ik.stop_mouse_aim(body.get('session_id'))
                elif route == '/api/ik/mouse-aim/pause':
                    if set(body)-{'session_id'}: raise ValueError('Mouse aim pause accepts a session ID only')
                    result = self.controller.ik.pause_mouse_aim(body.get('session_id'))
                elif route == '/api/ik/mouse-aim/heartbeat':
                    if set(body)-{'session_id', 'execution_id', 'plan_id'}: raise ValueError('Mouse aim heartbeat accepts session, execution and acknowledged plan IDs only')
                    result = self.controller.ik.mouse_aim_heartbeat(body.get('session_id'), body.get('execution_id'), body.get('plan_id'))
                elif route == '/api/ik/execute':
                    if set(body)-{'plan_id','speed','armed','frame_verified','scene_verified','session_id'}:
                        raise ValueError('Execute accepts a server plan ID and confirmation settings only')
                    speed = number(body.get('speed',30), 'IK SDK speed', 1, MAX_SDK_SPEED, True)
                    result = self.controller.ik.execute(body.get('plan_id'), speed, body.get('armed'), body.get('frame_verified'), body.get('scene_verified'), body.get('session_id'))
                elif route == '/api/ik/heartbeat': result = self.controller.ik.heartbeat(body.get('execution_id'))
                elif route == '/api/stop': result = self.controller.stop()
                else: return self.reply(404, {'error': 'Unknown endpoint'})
                if route not in ('/api/ik/plan','/api/ik/diffusion/plan','/api/ik/heartbeat', '/api/ik/mouse-aim/plan', '/api/ik/mouse-aim/heartbeat'): result = dict(result, hardware_enabled=self.hardware_enabled, mode_selectable=self.mode_selectable)
                self.reply(200, result)
        except (StaleMousePlan, diffusion.PlannerBusy) as exc: self.reply(409, {'error': str(exc)})
        except (ValueError, TypeError) as exc: self.reply(400, {'error': str(exc)})
        except Exception as exc: self.reply(503, {'error': str(exc)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--hardware', action='store_true', help='explicitly enable the existing hardware control API')
    mode.add_argument('--simulation-only', action='store_true', help='lock to simulation; disallow enabling hardware from the UI')
    args = parser.parse_args()
    if args.port != 8000: parser.error('The control bridge uses port 8000')
    socket.setdefaulttimeout(3)
    Handler.controller = RobotController()
    Handler.token = secrets.token_urlsafe(32)
    Handler.hardware_enabled = args.hardware
    Handler.mode_selectable = not args.simulation_only
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print('Robot web UI: http://127.0.0.1:8000 (or Vite at :5173); ' + ('hardware enabled' if args.hardware else 'SIMULATION ONLY'), flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        Handler.controller.shutdown()
        server.server_close()


if __name__ == '__main__': main()
