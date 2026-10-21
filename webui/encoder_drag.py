"""Bounded fixed-v2 encoder/speed experiment, not a velocity-mode driver.

Nominal scaling is 4096 encoder steps/revolution. Verify the centred per-joint
encoder direction against idle live angles before preparing any targets.
Speed words are interval-average step/s magnitudes, not measured velocities.
"""
import time
import numpy as np
from pymycobot.common import ProtocolCode
from pymycobot.error import calibration_parameters
from simulation.kinematics import ARM

STEPS_PER_DEG = 4096./360.
MAX_TRIAL_SPEED = 5
MAX_EXCURSION_DEG = 3.
MAX_WAYPOINTS = 20
MAX_CAPTURE_SECONDS = 2.


class EncoderDirectionChanged(RuntimeError):
    pass


def vector(value, name, integer=False):
    if (not isinstance(value, list) or len(value) != 6 or any(
            isinstance(x, bool) or not isinstance(x, (int, float))
            or not np.isfinite(x) or (integer and type(x) is not int) for x in value)):
        raise ValueError(f'Encoder trial requires six valid {name}')
    return np.asarray(value, float)


def direction_evidence(mapping):
    """Validate paired nonzero readings; a claimed sign alone is not evidence."""
    if mapping is None:return [None]*6
    evidence=mapping.get('encoder_direction_evidence')
    if evidence is None:
        angles=vector(mapping.get('start_deg'),'calibration angles')
        encoders=vector(mapping.get('start_encoders'),'integer calibration encoders',True)
        evidence=[]
        for angle,encoder in zip(angles,encoders):
            positive=abs(encoder-(2048.+angle*STEPS_PER_DEG))<=.5*STEPS_PER_DEG
            negative=abs(encoder-(2048.-angle*STEPS_PER_DEG))<=.5*STEPS_PER_DEG
            if not (positive or negative):raise ValueError('Encoder calibration readings are inconsistent')
            evidence.append(None if positive and negative else
                {'angle_deg':float(angle),'encoder':int(encoder),'direction':1 if positive else -1})
    if not isinstance(evidence,list) or len(evidence)!=6:
        raise ValueError('Encoder calibration requires six paired direction records')
    validated=[]
    for record in evidence:
        if record is None:validated.append(None);continue
        if not isinstance(record,dict) or set(record)!={'angle_deg','encoder','direction'}:
            raise ValueError('Invalid paired encoder calibration record')
        angle=record['angle_deg'];encoder=record['encoder'];direction=record['direction']
        if (isinstance(angle,bool) or not isinstance(angle,(int,float)) or not np.isfinite(angle)
                or type(encoder) is not int or not 1<=encoder<=4094
                or type(direction) is not int or direction not in (-1,1)):
            raise ValueError('Invalid paired encoder calibration values')
        positive=abs(encoder-(2048.+angle*STEPS_PER_DEG))<=.5*STEPS_PER_DEG
        negative=abs(encoder-(2048.-angle*STEPS_PER_DEG))<=.5*STEPS_PER_DEG
        if positive==negative or direction!=(1 if positive else -1):
            raise ValueError('Encoder direction calibration lacks an unambiguous paired reading')
        validated.append({'angle_deg':float(angle),'encoder':encoder,'direction':direction})
    return validated


def capture(robot, start_deg, *, verified_mapping=None):
    """Read only. Reject stale/moving or incompatible encoder telemetry."""
    started = time.monotonic()
    previous=direction_evidence(verified_mapping)
    start = vector(start_deg, 'starting angles')
    encoders = vector(robot.get_encoders(), 'integer encoder positions', True)
    after = vector(robot.get_angles(), 'joint angles')
    moving = robot.is_moving()
    if time.monotonic()-started > MAX_CAPTURE_SECONDS:
        raise RuntimeError('Encoder capture took over two seconds; no motion sent')
    if moving != 0 or np.max(np.abs(after-start)) > .1:
        raise RuntimeError('Arm moved during encoder capture; no motion sent')
    if np.any(encoders < 1) or np.any(encoders > 4094):
        raise ValueError('Encoder capture is outside the trial operating range')
    positive = np.abs(encoders-(2048.+after*STEPS_PER_DEG)) <= .5*STEPS_PER_DEG
    negative = np.abs(encoders-(2048.-after*STEPS_PER_DEG)) <= .5*STEPS_PER_DEG
    if not np.all(positive | negative):
        raise RuntimeError('Live encoders do not match either 2048-centred angle direction; trial disabled. '
            f'Encoders={encoders.astype(int).tolist()}, angles_deg={after.tolist()}, '
            f'minimum_residual_steps={np.round(np.minimum(np.abs(encoders-(2048.+after*STEPS_PER_DEG)), np.abs(encoders-(2048.-after*STEPS_PER_DEG))), 2).tolist()}')
    # Near zero both signs fit. Reuse only a previously unambiguous paired
    # reading; an unknown direction still permits stationary targets only.
    directions = np.where(positive & ~negative, 1, np.where(negative & ~positive, -1, 0))
    evidence=[];reused=[]
    for joint,observed in enumerate(directions):
        anchor=previous[joint]
        if anchor is not None and observed and observed!=anchor['direction']:
            raise EncoderDirectionChanged(f'Encoder direction changed for J{joint+1}; discard the previous calibration and plan again')
        if observed:
            anchor={'angle_deg':float(after[joint]),'encoder':int(encoders[joint]),'direction':int(observed)}
        elif anchor is not None:
            directions[joint]=anchor['direction'];reused.append(joint+1)
        evidence.append(anchor)
    return {'start_deg': after.tolist(), 'start_encoders': encoders.astype(int).tolist(),
            'encoder_directions': directions.tolist(),
            'encoder_direction_evidence':evidence,'reused_direction_joints':reused,
            'steps_per_degree': STEPS_PER_DEG,
            'capture_seconds': time.monotonic()-started,
            'scope': 'Nominal step/s conversion; physical speed and zero-speed behavior unqualified'}


def prepare(prepared, mapping, speed, *, maximum_waypoints=MAX_WAYPOINTS,
            maximum_excursion_deg=MAX_EXCURSION_DEG, maximum_speed_percent=MAX_TRIAL_SPEED):
    """Quantize and recheck all targets before a plan can be reviewed."""
    if speed > maximum_speed_percent:
        raise ValueError(f'Encoder trial requires arm speed at most {maximum_speed_percent}%; plan again')
    result = prepared['result']
    if not result.get('converged'):
        return
    waypoints = prepared['waypoints']
    if not waypoints:
        raise ValueError('Encoder route requires a target distinct from the current pose')
    if maximum_waypoints is not None and len(waypoints) > maximum_waypoints:
        raise ValueError(f'Encoder trial requires 1–{maximum_waypoints} waypoints; choose a small target')
    start = np.asarray(mapping['start_deg']); encoders = np.asarray(mapping['start_encoders'])
    directions = np.asarray(mapping['encoder_directions'])
    path = np.asarray(waypoints)
    unknown=np.flatnonzero((directions==0)&(np.max(np.abs(path-start),axis=0)>1e-6))
    if len(unknown):
        joints=', '.join(f'J{i+1}' for i in unknown)
        raise ValueError(f'A moving joint has an ambiguous encoder direction near zero ({joints}); no previously verified direction is available. Use a verified nonzero starting pose')
    if maximum_excursion_deg is not None and np.max(np.abs(path-start)) > maximum_excursion_deg:
        raise ValueError(f'Encoder trial is limited to {maximum_excursion_deg:g} degrees from the captured start on every joint')
    intervals = prepared['waypoint_intervals_s']
    if len(intervals) != len(waypoints):
        raise ValueError('Encoder trial needs a timed waypoint profile')
    counts = np.rint(encoders+(path-start)*STEPS_PER_DEG*directions).astype(int)
    if not np.any(counts != encoders):
        raise ValueError('Encoder trial target collapses at encoder resolution; choose a larger small target')
    if np.any(counts < 1) or np.any(counts > 4094):
        raise ValueError('Quantized encoder target is outside the trial operating range')
    quantized = start+(counts-encoders)/(STEPS_PER_DEG*np.where(directions == 0, 1, directions))
    if maximum_excursion_deg is not None and np.max(np.abs(quantized-start)) > maximum_excursion_deg:
        raise ValueError(f'Quantized encoder path exceeds the {maximum_excursion_deg:g} degree trial excursion')
    if np.any(np.deg2rad(quantized) < ARM.lower) or np.any(np.deg2rad(quantized) > ARM.upper):
        raise ValueError('Quantized encoder target exceeds a joint limit')
    from simulation.diffusion.env import ArmEnv
    with ArmEnv() as env:
        env.set_obstacle(None)
        if env.metadata() != prepared['environment'] or not env.path_valid(
                np.deg2rad(np.vstack([start, quantized])), .01):
            raise ValueError('Quantized encoder path fails the reviewed collision checks')
    packets = []
    previous = encoders
    # A zero speed word can have firmware-specific meaning. Stationary joints
    # retain their position target with a nonzero speed word of one instead.
    ceiling = int(np.floor(150.*speed/100.*STEPS_PER_DEG))
    for target, interval in zip(counts, intervals):
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError('Encoder waypoint interval must be positive')
        rates = np.maximum(1, np.rint(np.abs(target-previous)/interval).astype(int))
        if np.any(rates > ceiling):
            raise ValueError('Encoder quantization exceeds the trial speed allowance; choose another small target')
        packets.append({'encoders': target.tolist(), 'speeds': rates.tolist(),
                        'reference_interval_s': float(interval)})
        previous = target
    prepared.update(encoder_mapping=mapping, encoder_packets=packets)
    result['encoder_drag'] = {**mapping, 'command': 'set_encoders_drag',
        'maximum_excursion_deg': maximum_excursion_deg, 'maximum_speed_percent': maximum_speed_percent,
        'speed_scope': 'Nominal interval-average step/s; no signed velocity or acceleration command',
        'packets': packets, 'quantized_waypoints_deg': quantized.tolist()}


def verify(robot, start_deg, mapping):
    current = capture(robot, start_deg, verified_mapping=mapping)
    directions = np.asarray(mapping['encoder_directions'])
    observed_directions = np.asarray(current['encoder_directions'])
    if np.any((directions != 0) & (observed_directions != 0) & (directions != observed_directions)):
        raise RuntimeError('Encoder direction changed after planning; plan again')
    expected = np.asarray(mapping['start_encoders'])+(np.asarray(current['start_deg'])-
                np.asarray(mapping['start_deg']))*STEPS_PER_DEG*directions
    if np.max(np.abs(np.asarray(current['start_encoders'])-expected)) > 2:
        raise RuntimeError('Angle-to-encoder mapping changed after planning; plan again')


def send_once(robot, encoders, speeds):
    """The SDK set_encoders_drag packet, with asynchronous delivery/no retry.

The public SDK method lacks _async and its socket implementation can resend
while waiting for an ACK. Preserve its validator and encoder, send once.
"""
    calibration_parameters(class_name='MyCobot280Socket', encoders=encoders, speeds=speeds)
    robot._mesg(ProtocolCode.SET_ENCODERS_DRAG, encoders, speeds, _async=True)
