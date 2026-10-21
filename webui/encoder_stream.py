"""Sample checked fixed-v2 quintics for feedback-paced encoder replacements.

Every reviewed corner is retained. The executor stretches time when feedback
is late; it never skips samples or queues a burst to catch a preview clock.
"""
import copy
import numpy as np
from webui import encoder_drag

PERIOD_S = .2
MIN_SEGMENT_S = 2.
MAX_REFERENCE_SPEED_DEG_S = 1.
MAX_PACKETS = 200
MAX_DURATION_S = 60.
ARRIVAL_DEG = .75
MAX_LEAD_DEG = 1.
SETTLE_S = 1.
SETTLE_DRIFT_DEG = .1
MAX_STAGE_STEP_DEG = 2.


def prepare(prepared, mapping, speed, *, staged=False):
    result = prepared['result']
    if not result.get('converged'):
        return
    maximum_source = None if staged else encoder_drag.MAX_WAYPOINTS
    if not prepared['waypoints']:
        raise ValueError('Encoder stream requires a target distinct from the current pose')
    if maximum_source is not None and len(prepared['waypoints']) > maximum_source:
        raise ValueError(f'Encoder stream requires 1–{maximum_source} source waypoints; choose a smaller target')
    maximum_excursion = None if staged else encoder_drag.MAX_EXCURSION_DEG
    excursion = float(np.max(np.abs(np.asarray(prepared['waypoints'])-mapping['start_deg'])))
    if maximum_excursion is not None and excursion > maximum_excursion:
        raise ValueError(f'Encoder trial is limited to {maximum_excursion:g} degrees from the captured start on every joint; this route needs {excursion:.1f} degrees')
    result = prepared['result'] = copy.deepcopy(result)
    if staged:
        # Subdivide only already checked straight legs, retaining every
        # reviewed corner. Each new boundary is stopped and measured.
        from simulation.trajectory import joint_waypoints
        from webui.fixed_profile import retime_fixed
        original_segments = result['smoothing']['segments']
        vertices = [original_segments[0]['start_deg']]
        for segment in original_segments:
            vertices.extend(joint_waypoints(segment['start_deg'], [segment['end_deg']],
                                            MAX_STAGE_STEP_DEG, compress=False))
        result = prepared['result'] = retime_fixed(result, speed, path=np.deg2rad(vertices))
        result['smoothing']['collision_rechecked'] = True
    curve = result['smoothing']
    segments = curve['segments']
    if curve.get('method') != 'specification_checked_quintic' or not segments:
        raise ValueError('Encoder stream requires the checked fixed-v2 quintic profile')
    # Uniform stretching preserves the checked q/v/a joins and lowers every
    # derivative bound. Even a one-target plan now has a sampled approach.
    stretch = max(1., max(max(MIN_SEGMENT_S/s['duration_s'],
        1.875*np.max(np.abs(np.asarray(s['end_deg'])-s['start_deg'])) /
        (s['duration_s']*MAX_REFERENCE_SPEED_DEG_S)) for s in segments))
    if not staged and curve['duration_s']*stretch > MAX_DURATION_S:
        raise ValueError('Encoder stream exceeds 60 seconds; choose a shorter route')
    for frame in result['frames']:
        frame['time'] *= stretch
    curve['duration_s'] *= stretch
    curve['dt'] *= stretch
    for key, value in curve['motion_bounds'].items():
        curve['motion_bounds'][key] = value/stretch**(2 if 'acceleration' in key else 1)
    path, phases, velocities, stops = [], [], [], set()
    start = np.asarray(mapping['start_deg'])
    encoders = np.asarray(mapping['start_encoders'])
    directions = np.asarray(mapping['encoder_directions'])
    previous_counts = encoders
    for segment in segments:
        segment['start_time_s'] *= stretch
        segment['duration_s'] *= stretch
        segment['velocity_scale_rad_s'] /= stretch
        segment['velocity_allowance_scale'] /= stretch
        a = np.asarray(segment['start_deg']); delta = np.asarray(segment['end_deg'])-a
        duration = segment['duration_s']
        pieces = max(1, int(np.floor(duration/PERIOD_S+1e-9)))
        for i in range(1, pieces+1):
            u = i/pieces
            q = a+(10*u**3-15*u**4+6*u**5)*delta
            counts = np.rint(encoders+(q-start)*encoder_drag.STEPS_PER_DEG*directions).astype(int)
            corner = i == pieces
            if np.array_equal(counts, previous_counts) and not corner:
                continue
            path.append(q.tolist())
            phases.append(segment['start_time_s']+duration*u)
            velocities.append((30*u**2*(1-u)**2*delta/duration).tolist())
            previous_counts = counts
            if corner:
                stops.add(len(path)-1)
            if not staged and len(path) > MAX_PACKETS:
                raise ValueError('Encoder stream exceeds 200 packets; choose a shorter route')
    prepared.update(waypoints=path, stop_indices=stops,
                    waypoint_intervals_s=np.diff([0., *phases]).tolist())
    encoder_drag.prepare(prepared, mapping, speed, maximum_waypoints=None if staged else MAX_PACKETS,
        maximum_excursion_deg=maximum_excursion)
    prepared['waypoints'] = result['encoder_drag']['quantized_waypoints_deg']
    prepared['execution_mode'] = 'encoder_stream'
    for packet, phase, velocity in zip(prepared['encoder_packets'], phases, velocities):
        packet.update(reference_phase_s=float(phase), reference_velocity_deg_s=velocity)
    result['encoder_drag'].update(streaming=True, minimum_send_interval_s=PERIOD_S,
        staged=staged, maximum_packets=None if staged else MAX_PACKETS,
        maximum_reference_duration_s=None if staged else MAX_DURATION_S, stage_count=len(stops), stage_end_indices=sorted(stops),
        maximum_stage_step_deg=MAX_STAGE_STEP_DEG if staged else None,
        maximum_command_lead_deg=MAX_LEAD_DEG, arrival_deg=ARRIVAL_DEG,
        maximum_reference_speed_deg_s=MAX_REFERENCE_SPEED_DEG_S,
        settled_observation_s=SETTLE_S, settled_drift_deg=SETTLE_DRIFT_DEG, time_stretch=stretch,
        timing_scope='Feedback-paced replacements; no sample skipping or catch-up bursts. '
                     'Firmware interpolation and actual servo speeds remain unqualified.')


def paced_speeds(packet, previous_encoders, actual_interval):
    """Delay can lower a speed word, never raise it to recover elapsed time."""
    if not np.isfinite(actual_interval) or actual_interval <= 0:
        raise ValueError('Encoder stream needs a positive actual send interval')
    rates = np.maximum(1, np.rint(np.abs(np.asarray(packet['encoders'])-
        previous_encoders)/actual_interval).astype(int))
    return np.minimum(rates, packet['speeds']).tolist()
