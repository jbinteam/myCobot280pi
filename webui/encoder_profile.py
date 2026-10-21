"""Track the fixed-v2 time law with instantaneous six-joint speed words.

This is a position/speed adapter, not firmware velocity mode. Encoder targets
on the checked route bound travel; speed words use the nominal steps/s scale.
Fixed targets retain their reference clock. Mouse tracking paces progress from
live feedback because its objective is collision-free gaze, not time accuracy.
"""
import copy
import time
import numpy as np
from simulation.kinematics import ARM
from simulation.pose5 import axis_error
from simulation.reference_shape import evaluate as shape_evaluate
from webui.motion_settings import MAX_SDK_SPEED
from webui.robot_transport import reply_timeout
from webui import encoder_drag

PERIOD_S = .2
LOOKAHEAD_S = .2
TRACKING_DEG = 5.
# Retain one degree of headroom for the future target and encoder rounding.
LEAD_DEG = TRACKING_DEG + 1.
MOUSE_LEAD_DEG = 3.
ARRIVAL_DEG = .75
SETTLE_S = .5
MAX_CYCLE_S = 2.
STREAM_REPLY_TIMEOUT_S = .25
STREAM_QUERY_BUDGET_S = .5


def query_delay(last_send, query_cost_s, now):
    """Wake early enough to include measured query work in the next period."""
    return PERIOD_S if last_send is None else max(0., last_send+PERIOD_S-query_cost_s-now)


def send_deadline(last_send, query_done):
    return query_done if last_send is None else last_send+PERIOD_S


def feedback_phase(curve, phase, dt, live_deg, moving=False, hold=False):
    """Advance a mouse path only as fast as actual joints can follow it.

    Original checked corners remain barriers until the arm has reached them.
    A half-lead margin leaves room for the quantized lookahead command.
    """
    phase = min(max(0., phase), curve['duration_s'])
    if hold or not curve['segments']:
        return phase
    live = np.asarray(live_deg, float)
    corner = next((s for s in curve['segments']
                   if abs(phase-s['start_time_s']-s['duration_s']) < 1e-8), None)
    if corner is not None and (moving or np.max(np.abs(live-corner['end_deg'])) > ARRIVAL_DEG):
        return phase
    next_corner = next((s['start_time_s']+s['duration_s'] for s in curve['segments']
                        if s['start_time_s']+s['duration_s'] > phase+1e-8), curve['duration_s'])
    proposed = min(phase+max(0., min(dt, PERIOD_S*2)), next_corner, curve['duration_s'])
    limit = MOUSE_LEAD_DEG/2
    if np.max(np.abs(motion(curve, proposed)[0]-live)) <= limit:
        return proposed
    if np.max(np.abs(motion(curve, phase)[0]-live)) > limit:
        return phase
    lo, hi = phase, proposed
    for _ in range(32):
        at = (lo+hi)/2
        if np.max(np.abs(motion(curve, at)[0]-live)) <= limit:
            lo = at
        else:
            hi = at
    return lo


def motion(curve, phase):
    """Exact q, dq/dt, d²q/dt² of the reviewed shared quintic, in degrees."""
    if curve.get('method') == 'collision_checked_velocity_profile_c2':
        from simulation.profile_tracking import sample_motion
        q, v, a = sample_motion(curve, phase)
        index = next((i for i, s in enumerate(curve['segments'])
                      if phase < s['start_time_s']+s['duration_s']), max(0, len(curve['segments'])-1))
        return np.rad2deg(q), np.rad2deg(v), np.rad2deg(a), index
    segments = curve['segments']
    index = next((i for i, s in enumerate(segments)
                  if phase < s['start_time_s']+s['duration_s']), len(segments)-1)
    s = segments[index]
    u = float(np.clip((phase-s['start_time_s'])/s['duration_s'], 0., 1.))
    delta = np.asarray(s['end_deg'])-s['start_deg']
    if curve.get('reference_shape', {}).get('kind') == 'smooth_trapezoid':
        progress, rate, acceleration = shape_evaluate(u, curve['reference_shape'])
        return np.asarray(s['start_deg'])+progress*delta, rate*delta/s['duration_s'], acceleration*delta/s['duration_s']**2, index
    q = np.asarray(s['start_deg'])+(10*u**3-15*u**4+6*u**5)*delta
    v = 30*u**2*(1-u)**2*delta/s['duration_s']
    a = 60*u*(1-u)*(1-2*u)*delta/s['duration_s']**2
    return q, v, a, index


def packet(curve, phase, mapping, speed, *, live_deg=None, target_limit_s=None, maximum_lead_deg=LEAD_DEG):
    q, v, a, index = motion(curve, phase)
    corner = (curve['segments'][index]['start_time_s']+curve['segments'][index]['duration_s']
              if curve['segments'] else curve['duration_s'])
    # Never aim beyond the current checked corner, even on a delayed cycle.
    target_phase = min(phase+LOOKAHEAD_S, corner)
    if target_limit_s is not None:
        target_phase = min(target_phase, max(phase, target_limit_s))
    turn = next((t for t in curve.get('encoder_turns_s', ()) if t > phase+1e-9), None)
    if turn is not None:
        target_phase = min(target_phase, turn)
    start = np.asarray(mapping['start_deg']); directions = np.asarray(mapping['encoder_directions'])
    origin = np.asarray(mapping['start_encoders'])

    def quantize(at):
        target = motion(curve, at-1e-10)[0]
        if at >= curve['duration_s'] and curve['segments']:
            target = np.asarray(curve['segments'][-1]['end_deg'])
        counts = np.rint(origin+(target-start)*encoder_drag.STEPS_PER_DEG*directions).astype(int)
        quantized = start+(counts-origin)/(encoder_drag.STEPS_PER_DEG*np.where(directions == 0, 1, directions))
        return counts, quantized

    counts, quantized = quantize(target_phase)
    lead = None
    if live_deg is not None:
        live = np.asarray(live_deg, float)
        if live.shape != (6,) or not np.all(np.isfinite(live)):
            raise ValueError('Velocity-profile lookahead requires six finite live joint readings')
        lead = float(np.max(np.abs(quantized-live)))
        if lead > maximum_lead_deg:
            # Shorten only the future position target on the same checked leg.
            # The reference phase and analytic speed words remain unchanged.
            # If even q(t) is out of bounds, leave the existing tracking/lead
            # guards to stop execution rather than hiding reference lag.
            lo = min(phase, target_phase); hi = target_phase
            base_counts, base_quantized = quantize(lo)
            if np.max(np.abs(base_quantized-live)) <= maximum_lead_deg:
                counts, quantized = base_counts, base_quantized
                for _ in range(32):
                    at = (lo+hi)/2
                    candidate_counts, candidate_quantized = quantize(at)
                    if np.max(np.abs(candidate_quantized-live)) <= maximum_lead_deg:
                        lo = at; counts, quantized = candidate_counts, candidate_quantized
                    else:
                        hi = at
                target_phase = lo
                lead = float(np.max(np.abs(quantized-live)))
    if np.any(counts < 1) or np.any(counts > 4094) or np.any(np.deg2rad(quantized) < ARM.lower) or np.any(np.deg2rad(quantized) > ARM.upper):
        raise RuntimeError('Velocity-profile encoder target exceeds a joint/encoder limit')
    # Use the analytic derivative, not displacement divided by a send interval.
    # One is used with a bounded target because zero-word behaviour is unknown.
    ceiling = int(np.floor(150.*speed/100.*encoder_drag.STEPS_PER_DEG))
    words = np.maximum(1, np.floor(np.abs(v)*encoder_drag.STEPS_PER_DEG+1e-9).astype(int))
    if np.any(words > ceiling):
        raise RuntimeError('Velocity-profile speed exceeds the reviewed allowance')
    return {'encoders': counts.tolist(), 'speeds': words.tolist(),
            'target_deg': quantized.tolist(), 'reference_deg': q.tolist(),
            'reference_velocity_deg_s': v.tolist(), 'reference_acceleration_deg_s2': a.tolist(),
            'reference_phase_s': float(phase), 'segment_index': index,
            'target_phase_s': float(target_phase), 'lookahead_s_actual': max(0., target_phase-phase),
            'command_lead_deg': lead}


def prepare_mouse(plan, mapping, speed, *, deadline=None):
    """Check the entire quantized tracking curve, including moving joins."""
    from webui.tracking_scene import scene_for, step_for
    from simulation.diffusion.endpoint_selection import DeadlineEnv, SearchBudgetExpired
    def check():
        if deadline is not None and time.perf_counter() >= deadline:
            raise SearchBudgetExpired('Encoder candidate validation budget exhausted')
    check()
    encoder_drag.direction_evidence(mapping)
    result = plan['result']; frames = result['frames']; curve = plan['curve']
    if curve.get('boundary_blended'):
        # A moving C2 join can reverse within a leg. Do not command an encoder
        # position past that reversal while its instantaneous speed still has
        # the earlier direction. Compute derivative roots from exact pieces.
        from math import comb
        from numpy.polynomial import Polynomial
        from simulation.profile_tracking import pieces_for
        x = Polynomial([0., 1.]); turns = []
        for segment in curve['segments']:
            check()
            offset = segment['start_time_s']
            for controls, duration in pieces_for(np.deg2rad(segment['start_deg']), np.deg2rad(segment['end_deg']),
                    np.asarray(segment['start_velocity_rad_s']), np.asarray(segment['start_acceleration_rad_s2']),
                    segment['duration_s'], curve['reference_shape']):
                velocities = 5*np.diff(controls, axis=0)/duration
                for joint in range(6):
                    polynomial = sum((comb(4, k)*velocities[k, joint]*x**k*(1-x)**(4-k) for k in range(5)), Polynomial([0.]))
                    if np.max(np.abs(polynomial.coef)) < 1e-12:
                        continue
                    for root in polynomial.roots():
                        if abs(root.imag) < 1e-7 and 1e-9 < root.real < 1-1e-9:
                            turns.append(float(offset+root.real*duration))
                offset += duration
        curve['encoder_turns_s'] = sorted(set(turns))
    start = np.asarray(mapping['start_deg']); directions = np.asarray(mapping['encoder_directions'])
    origin = np.asarray(mapping['start_encoders']); path = np.asarray([f['q'] for f in frames])
    unknown = np.flatnonzero((directions == 0) & (np.max(np.abs(path-start), axis=0) > 1e-6))
    if len(unknown):
        raise ValueError('Moving joints have ambiguous encoder direction near zero: '+', '.join(f'J{i+1}' for i in unknown))
    quantized = []
    for frame in frames:
        check()
        cmd = packet(curve, frame['time'], mapping, speed, target_limit_s=frame['time'])
        quantized.append(cmd['target_deg'])
    with scene_for(plan) as env:
        env.set_obstacle(None)
        checked_env = env if deadline is None else DeadlineEnv(env, deadline)
        if env.metadata() != plan['environment'] or not checked_env.path_valid(np.deg2rad(quantized), step_for(env)):
            raise ValueError('Quantized mouse encoder curve fails pinned scene/visibility checks')
        if plan.get('visibility_object_m') is not None:
            result['quantized_visibility'] = env.audit(np.deg2rad(quantized), .005)
    check()
    endpoints = [packet(curve, s['start_time_s']+s['duration_s'], mapping, speed,
                        target_limit_s=s['start_time_s']+s['duration_s'])['target_deg'] for s in curve['segments']]
    if plan.get('tracking_region') is not None and result.get('converged'):
        from webui.tracking_scene import region_acceptance
        acceptance = region_acceptance(np.deg2rad(quantized[-1]), plan, 5., 3.)
        if not acceptance['accepted']:
            raise ValueError('Quantized mouse endpoint is outside the tracking region')
        result['quantized_tracking_region_acceptance'] = acceptance
    plan.update(encoder_mapping=copy.deepcopy(mapping), waypoints=endpoints or [quantized[-1]],
                execution_mode='encoder_feedback_tracking', joint_command='encoder_drag_profile', execution_timeout_s=180.)
    result.update(execution_mode='encoder_feedback_tracking', joint_command='encoder_drag_profile',
                  velocity_shape=curve['reference_shape'])
    result['encoder_drag'] = {**copy.deepcopy(mapping), 'command':'set_encoders_drag',
        'velocity_profile':True, 'streaming':True, 'staged':False, 'stage_count':0,
        'minimum_send_interval_s':PERIOD_S, 'lookahead_s':LOOKAHEAD_S,
        'maximum_tracking_error_deg':None, 'maximum_command_lead_deg':MOUSE_LEAD_DEG,
        'arrival_deg':ARRIVAL_DEG, 'maximum_speed_percent':MAX_SDK_SPEED,
        'quantized_waypoints_deg':plan['waypoints'],
        'speed_scope':'Instantaneous analytic dq/dt converted to nominal step/s magnitudes; direction comes from encoder targets',
        'timing_scope':'Live-feedback paced mouse tracking; trajectory deviation is diagnostic, each actual-to-command connection is collision checked'}


def prepare(prepared, mapping, speed):
    result = prepared['result']
    if not result.get('converged'):
        return
    result = prepared['result'] = copy.deepcopy(result)
    curve = result['smoothing']
    if curve.get('method') not in ('specification_checked_quintic', 'specification_checked_velocity_shape') or not curve.get('segments'):
        raise ValueError('Velocity profile requires the checked v2 timed trajectory')
    # Give every original leg at least four nominal updates. A common time
    # stretch preserves q/v/a joins and relative timing without adding stops.
    stretch = max(1., 4*PERIOD_S/min(s['duration_s'] for s in curve['segments']))
    for frame in result['frames']:
        frame['time'] *= stretch
    curve['duration_s'] *= stretch
    curve['dt'] *= stretch
    for key, value in curve['motion_bounds'].items():
        curve['motion_bounds'][key] = value/stretch**(2 if 'acceleration' in key else 1)
    for segment in curve['segments']:
        segment['start_time_s'] *= stretch
        segment['duration_s'] *= stretch
        segment['velocity_scale_rad_s'] /= stretch
        segment['velocity_allowance_scale'] /= stretch
    # Validate the encoder mapping and quantized original corners. Do not add
    # stages or replace the curve with rest-to-rest samples.
    prepared['waypoints'] = [s['end_deg'] for s in curve['segments']]
    prepared['stop_indices'] = set(range(len(prepared['waypoints'])))
    prepared['waypoint_intervals_s'] = [s['duration_s'] for s in curve['segments']]
    encoder_drag.prepare(prepared, mapping, speed, maximum_waypoints=None, maximum_excursion_deg=None,
                         maximum_speed_percent=MAX_SDK_SPEED)
    prepared['execution_mode'] = 'encoder_profile'
    prepared['waypoints'] = result['encoder_drag']['quantized_waypoints_deg']
    result['encoder_drag'].update(velocity_profile=True, streaming=True, staged=False,
        stage_count=0, minimum_send_interval_s=PERIOD_S, lookahead_s=LOOKAHEAD_S, lookahead_limited_by_live_lead=True,
        cadence_time_stretch=stretch,
        maximum_tracking_error_deg=TRACKING_DEG, maximum_command_lead_deg=LEAD_DEG,
        arrival_deg=ARRIVAL_DEG, speed_scope='Instantaneous analytic dq/dt converted to nominal step/s magnitudes; direction comes from encoder targets',
        timing_scope='Telemetry starts before each 0.2-second send deadline using measured query cost. Monotonic trajectory clock; no waypoint arrival waits or added 2-degree stops. Lag beyond tolerance sends Stop. Physical speed units and timing remain unqualified.')
    # These packets belong to the old waypoint executor, not this time sampler.
    prepared['encoder_packets'] = None
    result['encoder_drag'].pop('packets', None)


def run(owner, plan, speed, generation, *, mouse=None):
    from webui.tracking_scene import scene_for, step_for
    from webui.hardware_ik import HEARTBEAT_TIMEOUT, FINAL_ERROR_MM, FINAL_ORIENTATION_DEG
    c = owner.controller
    curve = plan['result']['smoothing']
    waiting_at = time.monotonic(); started = None; last_send = None
    last_segment = 0; settled_at = None; settled_pose = None
    approach_holding = False
    query_cost_s = 0.
    feedback_at = None
    cycle_at = time.monotonic()
    scene = None
    try:
        scene = scene_for(plan); scene.set_obstacle(None)
        if scene.metadata() != plan['environment']:
            raise RuntimeError('Velocity-profile scene differs from the reviewed v2 scene')
        # Start the next telemetry cycle early enough to finish by the send
        # deadline. Waiting a full period before queries adds their latency to
        # every command interval. Overruns use actual time, without bursts.
        while not c.closed.wait(query_delay(last_send, query_cost_s, time.monotonic())):
            with c.lock:
                if c.generation != generation or c.motion != 'ik' or not c.robot:
                    return
                cycle_at = time.monotonic()
                if cycle_at-owner.heartbeat_at > HEARTBEAT_TIMEOUT:
                    raise RuntimeError('Browser heartbeat lost')
                budget_start = started if mouse is not None and started is not None else waiting_at
                if cycle_at-budget_start > plan['execution_timeout_s']:
                    raise RuntimeError('Velocity-profile execution exceeded its route budget')
                if not owner.heartbeat_confirmed:
                    continue
                with reply_timeout(c.robot, STREAM_REPLY_TIMEOUT_S, budget_s=STREAM_QUERY_BUDGET_S):
                    c._read(); c._health()
                    if mouse is not None and started is None:
                        owner.enable_refresh_mode()
                    refresh_mode = owner.read_refresh_mode()
                if refresh_mode != 1:
                    raise RuntimeError('Velocity-profile execution lost refresh mode 1')
                now = time.monotonic()
                if now-cycle_at > MAX_CYCLE_S or (last_send is not None and now-last_send > MAX_CYCLE_S):
                    raise RuntimeError('Velocity-profile telemetry/send cycle exceeded two seconds')
                if not scene.valid(np.deg2rad(c.angles)):
                    raise RuntimeError('Live pose fails reviewed collision/object visibility checks')
                query_done = time.monotonic()
                query_cost_s = query_done-cycle_at
                deadline = send_deadline(last_send, query_done)
                deadline_wait_s = max(0., deadline-query_done)
            # Release the controller lock while waiting: Stop and browser
            # heartbeats must be able to act between telemetry and delivery.
            if c.closed.wait(deadline_wait_s):
                return
            with c.lock:
                if c.generation != generation or c.motion != 'ik' or not c.robot:
                    return
                now = time.monotonic()
                if now-owner.heartbeat_at > HEARTBEAT_TIMEOUT:
                    raise RuntimeError('Browser heartbeat lost before velocity-profile delivery')
                if now-cycle_at > MAX_CYCLE_S or (last_send is not None and now-last_send > MAX_CYCLE_S):
                    raise RuntimeError('Velocity-profile telemetry/send cycle exceeded two seconds')
                budget_start = started if mouse is not None and started is not None else waiting_at
                if now-budget_start > plan['execution_timeout_s']:
                    raise RuntimeError('Velocity-profile execution exceeded its route budget')
                if started is None:
                    if c.moving:
                        raise RuntimeError('Unplanned motion before velocity-profile execution')
                    started = now
                    if mouse is not None:
                        mouse.clock_origin = started
                if mouse is not None:
                    mouse.observe_live(c.angles, now)
                    previous_id = plan['id']
                    plan, started = mouse.advance_pending(plan, started, now, scene)
                    curve = plan['curve']
                    if plan['id'] != previous_id:
                        last_segment = 0; settled_at = settled_pose = None
                        approach_holding = False
                        feedback_at = now
                elapsed = now-started
                phase = min(elapsed, curve['duration_s'])
                paused = False
                if mouse is not None:
                    dt = 0. if feedback_at is None else now-feedback_at
                    previous_phase = mouse.phase
                    held = mouse.feedback_hold(now)
                    phase = feedback_phase(curve, previous_phase, dt, c.angles, c.moving, held)
                    paused = held or (dt > 0 and phase < min(previous_phase+dt, curve['duration_s'])-1e-6)
                    feedback_at = now
                limit = mouse.pending_join_limit() if mouse is not None else None
                cmd = packet(curve, phase, plan['encoder_mapping'], speed, live_deg=c.angles, target_limit_s=limit,
                             maximum_lead_deg=MOUSE_LEAD_DEG if mouse is not None else LEAD_DEG)
                if mouse is not None:
                    mouse.phase = phase
                index = cmd['segment_index']
                if mouse is None and index > last_segment:
                    corner = curve['segments'][last_segment]['end_deg']
                    if index > last_segment+1 or np.max(np.abs(np.asarray(c.angles)-corner)) > ARRIVAL_DEG:
                        raise RuntimeError('Velocity-profile transport/tracking missed a checked corner')
                last_segment = index
                gap = float(np.max(np.abs(np.asarray(c.angles)-cmd['reference_deg'])))
                if mouse is None and gap > TRACKING_DEG:
                    raise RuntimeError(f'Velocity-profile tracking error {gap:.2f}° exceeds {TRACKING_DEG:g}°; reference timing retained')
                if np.max(np.abs(np.asarray(c.angles)-cmd['target_deg'])) > (MOUSE_LEAD_DEG if mouse is not None else LEAD_DEG)+1e-8:
                    raise RuntimeError('Velocity-profile command lead exceeds its reviewed allowance')
                if not scene.path_valid(np.deg2rad([c.angles, cmd['target_deg']]), step_for(scene)):
                    raise RuntimeError('Velocity-profile live-to-target segment fails collision/object visibility checks')
                position = ARM.fk(np.deg2rad(c.angles))*1000
                error = float(np.linalg.norm(position-plan['target_mm']))
                orientation = axis_error(ARM.transform(np.deg2rad(c.angles))[:3,2], np.asarray(plan['target_axis']))[2]
                region_acceptance = None
                if mouse is not None and plan.get('tracking_region') is not None:
                    from webui.tracking_scene import region_acceptance as check_region
                    distance_tolerance = mouse.approach['distance_tolerance_mm'] if mouse.approach else FINAL_ERROR_MM
                    region_acceptance = check_region(np.deg2rad(c.angles), plan, distance_tolerance, FINAL_ORIENTATION_DEG)
                sample = {'time': elapsed, 'trajectory_elapsed_s':elapsed, 'q': c.angles[:], 'position_mm': position.tolist(),
                          'error_mm': error, 'orientation_error_deg': orientation,
                          'tracking_error_deg': gap, **{k: cmd[k] for k in (
                              'reference_deg', 'reference_velocity_deg_s', 'reference_acceleration_deg_s2', 'reference_phase_s',
                              'target_phase_s', 'lookahead_s_actual', 'command_lead_deg')}}
                if mouse is not None:
                    from simulation.diffusion.tracking import cursor_metrics
                    gaze = cursor_metrics(np.deg2rad(c.angles), np.asarray(plan['cursor_mm']), 5., 3.)
                    nominal = motion(curve, min(elapsed, curve['duration_s']))[0]
                    sample.update(time=now-waiting_at, active_plan_id=plan['id'], max_joint_error_deg=gap,
                        reference_lag_s=max(0., elapsed-phase),
                        nominal_tracking_error_deg=float(np.max(np.abs(np.asarray(c.angles)-nominal))),
                        cursor_direction_error_deg=gaze['direction_error_deg'],
                        aim_error_deg=gaze['direction_error_deg'],
                        tracking_paused=paused, self_collision_checked=True)
                    if region_acceptance is not None:
                        sample['tracking_region_acceptance'] = region_acceptance
                        owner.status.update(tracking_region_acceptance=region_acceptance,
                            cursor_distance_mm=region_acceptance['distance_mm'],
                            cursor_direction_error_deg=region_acceptance['direction_error_deg'])
                    owner.status.update(curve_time_s=phase, curve_duration_s=curve['duration_s'],
                                        velocity_limit_deg_s=float(np.rad2deg(mouse.speed_rad_s)),
                                        reference_lag_s=sample['reference_lag_s'],
                                        cursor_direction_error_deg=sample['cursor_direction_error_deg'],
                                        aim_error_deg=sample['aim_error_deg'],
                                        nominal_tracking_error_deg=sample['nominal_tracking_error_deg'],
                                        tracking_pause_reason=('planning_latest_cursor' if held else 'waiting_for_live_arm' if paused else None),
                                        self_collision_checked=True)
                owner.status['samples'] = (owner.status['samples']+[sample])[-1000:]
                owner.status.update(estimated_error_mm=error, estimated_orientation_error_deg=orientation,
                    tracking_error_deg=gap, reference_phase_s=phase, completed_waypoints=index,
                    tracking_paused=paused, command_deg=cmd['target_deg'])
                if (mouse is not None and mouse.approach and region_acceptance is not None
                        and region_acceptance['accepted'] and not owner.mouse_pending_plan and not held):
                    # End this stage at a measured acceptable pose, even before
                    # the planned endpoint. Keep the session/lease and recheck
                    # the band after Stop; coasting out resumes the checked path.
                    if not approach_holding:
                        from pymycobot.common import ProtocolCode
                        c._send_motion(lambda: c.robot._mesg(ProtocolCode.STOP, _async=True))
                        approach_holding = True
                        settled_at = settled_pose = None
                    if c.moving:
                        settled_at = settled_pose = None
                    elif settled_at is None or np.max(np.abs(np.asarray(c.angles)-settled_pose)) > .1:
                        settled_at, settled_pose = now, np.asarray(c.angles)
                    owner.status.update(approach_distance_tolerance_mm=mouse.approach['distance_tolerance_mm'],
                        message='Holding inside approach distance allowance')
                    if settled_at is not None and now-settled_at >= SETTLE_S:
                        c.motion = c.target = None
                        owner.status.update(state='completed', final_goal_accepted=True,
                            cursor_acceptance=region_acceptance, tracking_accepted=True,
                            replan_required=False, feedback_replan_required=False,
                            approach_completed_with_allowance=True,
                            message='Approach region reached within distance allowance')
                        return
                    last_send = now
                    continue
                if approach_holding:
                    approach_holding = False
                    settled_at = settled_pose = None
                if phase >= curve['duration_s']:
                    endpoint_error = float(np.max(np.abs(np.asarray(c.angles)-plan['waypoints'][-1])))
                    reached = endpoint_error <= ARRIVAL_DEG and not c.moving
                    if not reached:
                        settled_at = settled_pose = None
                    elif settled_at is None or np.max(np.abs(np.asarray(c.angles)-settled_pose)) > .1:
                        settled_at, settled_pose = now, np.asarray(c.angles)
                    if reached and now-settled_at >= SETTLE_S:
                        if mouse is not None and owner.mouse_pending_plan:
                            continue
                        if mouse is None and (error > FINAL_ERROR_MM or orientation > FINAL_ORIENTATION_DEG):
                            raise RuntimeError('Stopped velocity-profile endpoint misses Cartesian/direction tolerance')
                        correction = False
                        if mouse is not None and plan['result']['converged']:
                            from simulation.diffusion.tracking import cursor_metrics
                            acceptance = region_acceptance if region_acceptance is not None else cursor_metrics(np.deg2rad(c.angles), np.asarray(plan['cursor_mm']), 5., 3.)
                            correction = not acceptance['accepted']
                            owner.status.update(cursor_acceptance=acceptance, final_goal_accepted=not correction)
                            sample['cursor_acceptance'] = acceptance
                            if correction:
                                owner.status.update(replan_required=True, feedback_replan_required=True)
                        if mouse is None:
                            owner.release_mouse_mode()
                        c.motion = c.target = None
                        owner.status.update(state='completed', completed_waypoints=len(plan['waypoints']),
                            message='Checked path completed; cursor viewpoint needs a fresh live correction' if correction
                                    else 'Velocity profile completed with stopped endpoint; error is estimated from URDF FK')
                        return
                    if mouse is None and elapsed > curve['duration_s']+10.:
                        raise RuntimeError('Velocity-profile endpoint did not settle within ten seconds')
                # Recheck time/lease after blocking collision and health work.
                send_at = time.monotonic()
                if send_at-cycle_at > MAX_CYCLE_S or send_at-owner.heartbeat_at > HEARTBEAT_TIMEOUT:
                    raise RuntimeError('Velocity-profile command became stale before delivery')
                if send_at-now > PERIOD_S:
                    raise RuntimeError('Velocity-profile validation exceeded the sampling period')
                c._send_motion(lambda: encoder_drag.send_once(c.robot, cmd['encoders'], cmd['speeds']))
                details = {'sent_encoder_targets': cmd['encoders'], 'sent_encoder_speeds': cmd['speeds'],
                           'actual_send_interval_s': None if last_send is None else send_at-last_send,
                           'telemetry_cycle_s': query_cost_s, 'deadline_wait_s': deadline_wait_s,
                           'send_deadline_lateness_s': max(0., send_at-deadline),
                           'encoder_speed_units': 'nominal step/s', 'joint_command': plan['joint_command'],
                           'target_phase_s': cmd['target_phase_s'], 'lookahead_s_actual': cmd['lookahead_s_actual'],
                           'command_lead_deg': cmd['command_lead_deg']}
                sample.update(details); owner.status.update(details)
                c.target = cmd['target_deg']; last_send = send_at
    except Exception as exc:
        with c.lock:
            if c.generation != generation or c.motion != 'ik':
                return
            reader = getattr(c.robot, '_read', None)
            queries = [q for q in getattr(reader, 'recent_queries', ())
                       if q.get('started_at', 0.) >= cycle_at]
            owner.status['failed_cycle_timing'] = {
                'cycle_elapsed_s': time.monotonic()-cycle_at,
                'since_last_send_s': None if last_send is None else time.monotonic()-last_send,
                'queries': [dict(q) for q in queries]}
            try:
                c._stop()
            except Exception:
                c._close()
            c.error = f'Velocity-profile execution stopped: {exc}. Plan again before retrying.'
            owner.status.update(state='failed', message=c.error)
    finally:
        if scene is not None:
            scene.close()
