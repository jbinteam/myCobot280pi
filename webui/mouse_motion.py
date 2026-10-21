"""Bounded commands along a checked polyline, with mandatory corner stops."""
import numpy as np
from simulation.trajectory import joint_waypoints
from simulation.motion_limits import envelope, scale_needed

MAX_VELOCITY_DEG_S = 20.
MAX_ACCELERATION_DEG_S2 = 60.
MAX_REFERENCE_LEAD_DEG = 3.
MAX_LIVE_COMMAND_DEG = 4.


class StaleMousePlan(RuntimeError):
    """A moving start needs a new plan; retain the current checked route."""


class JointServo:
    """Feedback-limited velocity integrated into persistent joint references.

    SDK speed remains a motor setting. These limits bound the requested
    reference profile; physical servo speed/acceleration are not guaranteed.
    A persistent reference crosses servo deadband without unbounded lead.
    """
    def __init__(self, start_deg, speed=30):
        self.reference = np.asarray(start_deg, float).copy()
        self.velocity = np.zeros(6)
        self.set_speed(speed)

    def set_speed(self, speed):
        self.max_velocity = MAX_VELOCITY_DEG_S*speed/30
        # Keep acceleration fixed during retargeting so existing velocities
        # decelerate smoothly even if a new plan requests a lower SDK speed.
        self.max_acceleration = MAX_ACCELERATION_DEG_S2

    def step(self, live_deg, target_deg, dt):
        live, target = np.asarray(live_deg, float), np.asarray(target_deg, float)
        if (live.shape != (6,) or target.shape != (6,) or not np.isfinite(live).all()
                or not np.isfinite(target).all() or not np.isfinite(dt) or not 0 < dt <= .15):
            raise ValueError('Expected finite live joints, servo target and a 0–150 ms time step')
        delta = target-self.reference
        distance = float(np.max(np.abs(delta)))
        headroom = max(0., MAX_REFERENCE_LEAD_DEG-float(np.max(np.abs(self.reference-live))))
        def brake_speed(remaining):
            a = self.max_acceleration
            return max(0., np.sqrt((a*dt)**2+2*a*remaining)-a*dt)
        magnitude = min(self.max_velocity, 6*float(np.max(np.abs(target-live))),
                        brake_speed(distance), brake_speed(headroom))
        desired = delta/distance*magnitude if distance > 1e-10 else np.zeros(6)
        # The per-joint servo cap alone does not bound combined flange motion.
        desired /= scale_needed(envelope(np.deg2rad(np.abs(desired)), np.zeros(6)))
        change = desired-self.velocity
        largest = float(np.max(np.abs(change)))
        velocity = self.velocity+change*min(1., self.max_acceleration*dt/largest) if largest else desired
        def permitted(fraction):
            candidate = self.velocity+fraction*(velocity-self.velocity)
            return scale_needed(envelope(np.deg2rad(np.maximum(np.abs(candidate), np.abs(self.velocity))),
                np.deg2rad(np.abs(candidate-self.velocity)/dt))) <= 1.+1e-8
        if not permitted(1.):
            lo, hi = 0., 1.
            for _ in range(24):
                mid = (lo+hi)/2
                if permitted(mid): lo = mid
                else: hi = mid
            velocity = self.velocity+lo*(velocity-self.velocity)
        reference = self.reference+velocity*dt
        if np.max(np.abs(reference-live)) > MAX_LIVE_COMMAND_DEG+1e-8:
            raise RuntimeError('Live tracking drift exceeds the servo command bound')
        self.reference, self.velocity = reference, velocity
        return reference.copy()

    def at_target(self, target_deg):
        return (np.max(np.abs(self.reference-target_deg)) <= .01
                and np.max(np.abs(self.velocity)) <= .1)


def rebase_path(reference_deg, live_deg, path_deg, space):
    """Join a nearby checked route without replaying its moving-start prefix.

    The join is within one degree of the current reference; both bridges are
    checked. The following original route vertices remain mandatory stops.
    Returns (targets, stop indices, maximum join displacement in degrees).
    """
    reference, live, path = (np.asarray(v, float) for v in (reference_deg,live_deg,path_deg))
    if (path.ndim != 2 or path.shape[1:] != (6,) or not len(path)
            or not np.isfinite(path).all()):
        raise ValueError('Expected a finite checked mouse route')
    candidates = []
    if len(path) == 1:
        candidates.append((float(np.max(np.abs(path[0]-reference))),0,path[0]))
    for index,(a,b) in enumerate(zip(path,path[1:])):
        delta = b-a
        fraction = np.clip(np.dot(reference-a,delta)/np.dot(delta,delta),0,1) if np.dot(delta,delta) else 0
        join = a+fraction*delta
        candidates.append((float(np.max(np.abs(join-reference))),index+1,join))
    for distance,index,join in sorted(candidates,key=lambda c:c[0]):
        if distance > 1. or np.max(np.abs(join-live)) > MAX_LIVE_COMMAND_DEG:
            continue
        if not (space.edge_valid(np.deg2rad(reference),np.deg2rad(join),.04)
                and space.edge_valid(np.deg2rad(live),np.deg2rad(join),.04)):
            continue
        targets = path[index:].tolist()
        stops = set(range(len(targets)))
        if distance > .01:
            targets.insert(0,join.tolist())
            stops = {i+1 for i in stops}
        if not targets:
            targets = [join.tolist()]
        if len(targets) == 1:
            stops.add(0)
        return targets,stops,distance
    raise StaleMousePlan('No nearby route join passes live pose and self-collision checks; plan the latest cursor again')


def command_path(start_deg, path_deg, space, max_step_deg=3., edge_step_rad=.04):
    """Resample path legs, preserving exact vertices rather than preview ticks.

    Each leg is rechecked, and its final command must settle before turning.
    Interior commands may advance while moving, after near-arrival telemetry.
    Returns commands and the zero-based indices requiring a stopped arm.
    """
    start = np.asarray(start_deg, float)
    path = np.asarray(path_deg, float)
    if (start.shape != (6,) or path.ndim != 2 or path.shape[1:] != (6,)
            or not len(path) or not np.isfinite(start).all() or not np.isfinite(path).all()
            or not np.allclose(path[0], start, atol=1e-8, rtol=0)):
        raise ValueError('Hardware mouse path must begin at the live six-joint start')
    if not space.valid(np.deg2rad(start)):
        raise ValueError('Hardware mouse path has an invalid live start')
    commands, stops = [], set()
    previous = start
    for vertex in path[1:]:
        if np.array_equal(previous, vertex):
            continue
        if not space.edge_valid(np.deg2rad(previous), np.deg2rad(vertex), edge_step_rad):
            raise ValueError('Hardware mouse path segment violates joint limits or self-collision clearance')
        commands.extend(joint_waypoints(previous, [vertex], max_step_deg, compress=False))
        stops.add(len(commands)-1)
        previous = vertex
    return commands, stops
