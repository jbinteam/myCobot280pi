"""Bounded firmware qualification for directional jog; no trajectory substitution.

A JOG_ANGLE packet addresses one joint. Qualify whether a later packet for a
different joint leaves the first moving before considering coordinated jogging.
SDK units are deliberately never reported as calibrated physical velocities.
"""
import time
import numpy as np
from pymycobot.common import ProtocolCode
from simulation.kinematics import ARM
from webui.hardware_diffusion import verified_status

JOINTS=(4,6)
SDK_SPEED=1
PHASE_SECONDS=.8
MAX_EXCURSION_DEG=2.5
CHECK_SPAN_DEG=4.


def verify_execution_ready():
    raise ValueError('Directional jog execution is disabled: the Atom 6.5 trial exceeded its travel guard while telemetry delayed Stop. A Pi-side watchdog and firmware command validation are required before another trial.')


def prepare(start_deg,body):
    if body:raise ValueError('Jog qualification uses fixed J4/J6, SDK speed 1 and bounded durations; accepts no overrides')
    start=np.asarray(start_deg,float)
    signs=[1 if start[j-1]<(np.rad2deg(ARM.lower[j-1])+np.rad2deg(ARM.upper[j-1]))/2 else -1 for j in JOINTS]
    status=verified_status()
    from simulation.diffusion.env import ArmEnv
    with ArmEnv() as env:
        env.set_obstacle(None)
        if env.metadata()!=status['environment']:raise RuntimeError('Jog collision scene differs from the reviewed table scene')
        # Sample the complete two-joint rectangle, including reverse drift and
        # a stopping margin beyond the monitored 2.5-degree excursion.
        for a in np.arange(-CHECK_SPAN_DEG,CHECK_SPAN_DEG+.1,.5):
            for b in np.arange(-CHECK_SPAN_DEG,CHECK_SPAN_DEG+.1,.5):
                q=start.copy();q[JOINTS[0]-1]+=a;q[JOINTS[1]-1]+=b
                if not env.valid(np.deg2rad(q)):
                    raise ValueError('Jog qualification rectangle fails self/table/joint-limit validation; use a clearer pose')
    t=ARM.transform(np.deg2rad(start));xyz=(t[:3,3]*1000).tolist();axis=t[:3,2].tolist()
    diagnostic={'variant':'jog_qualification','joint_ids':list(JOINTS),'directions':[int(s>0) for s in signs],
        'sdk_speed':SDK_SPEED,'phase_seconds':PHASE_SECONDS,'maximum_excursion_deg':MAX_EXCURSION_DEG,
        'maximum_step_deg':0.,'scope':'SDK direction-and-speed test; physical velocity mapping uncalibrated'}
    result={'method':'jog_qualification','task':'pose5','converged':True,'frames':[{'q':start.tolist(),'time':0.}],
        'target_mm':xyz,'target_axis':axis,'final_error_mm':0.,'final_orientation_error_deg':0.,
        'planned_sdk_speed':SDK_SPEED,'diagnostic':diagnostic,'checkpoint_sha256':status['checkpoint_sha256'],
        'smoothing':{'duration_s':2*PHASE_SECONDS}}
    return {'result':result,'target_mm':xyz,'eef_rotation':t[:3,:3].tolist(),'waypoints':[],
        'stop_indices':set(),'environment':status['environment'],'execution_mode':'jog_qualification'}


def analyze(samples,events,start):
    # Avoid the first .15 s after either command to reduce transition aliasing.
    a,b,stop=events
    phases=[]
    for begin,end in ((a['time_s'],b['time_s']),(b['time_s'],stop['time_s'])):
        points=[x for x in samples if begin+.15<=x['time']<=end]
        if len(points)<2 or points[-1]['time']<=points[0]['time']:
            phases.append(None);continue
        dt=points[-1]['time']-points[0]['time']
        delta=(np.array(points[-1]['q'])-points[0]['q'])
        phases.append({'time_span_s':dt,'travel_deg':delta.tolist(),
            'mean_velocity_deg_s':(delta/dt).tolist(),'sample_count':len(points)})
    conclusion='inconclusive'
    if all(phases):
        first,both=phases
        old=abs(first['travel_deg'][JOINTS[0]-1]);new=abs(both['travel_deg'][JOINTS[0]-1]);other=abs(both['travel_deg'][JOINTS[1]-1])
        if old>=.18 and other>=.18:
            ratio=abs(both['mean_velocity_deg_s'][JOINTS[0]-1])/abs(first['mean_velocity_deg_s'][JOINTS[0]-1])
            if new>=.18 and ratio>=.5:conclusion='both_joints_observed_moving_after_second_jog'
            elif new<=.09 and ratio<=.2:conclusion='first_joint_stopped_after_second_jog'
    return {'conclusion':conclusion,'phases':phases,
        'scope':'Single low-speed firmware trial; not six-axis qualification or a calibrated SDK-to-deg/s mapping',
        'final_delta_deg':(np.asarray(samples[-1]['q'])-start).tolist()}


def run(owner,plan,generation):
    c=owner.controller;start=np.array(plan['start_deg']);began=time.monotonic()
    phase='waiting';phase_at=None;events=[];settings=[0]*6;started=False;stopped=False;last_q=start.copy();last_time=None
    from simulation.diffusion.env import ArmEnv
    scene=None
    try:
        scene=ArmEnv();scene.set_obstacle(None)
        if scene.metadata()!=plan['environment']:raise RuntimeError('Jog scene changed after review')
        while not c.closed.wait(.05):
            with c.lock:
                if c.generation!=generation or c.motion!='ik' or not c.robot:return
                now=time.monotonic()
                if now-owner.heartbeat_at>5:raise RuntimeError('Jog browser heartbeat lost')
                if now-began>10:raise RuntimeError('Jog qualification exceeded its deadline')
                if not owner.heartbeat_confirmed:continue
                c._read();now=time.monotonic();q=np.array(c.angles);delta=q-start
                if started and not stopped and now-events[0]['time_s']-began>=2.5:
                    raise RuntimeError('Jog deadline elapsed during telemetry; no further jog command allowed')
                if not scene.valid(np.deg2rad(q)):raise RuntimeError('Jog live pose fails self/table/limit collision checks')
                if owner.read_refresh_mode()!=1:raise RuntimeError('Jog qualification lost refresh mode 1')
                if any(abs(delta[j-1])>MAX_EXCURSION_DEG for j in JOINTS):raise RuntimeError('Jog excursion limit reached')
                if any(abs(delta[j])>.5 for j in range(6) if j+1 not in JOINTS):raise RuntimeError('An untested joint moved during jog qualification')
                if last_time is not None:
                    velocity=np.abs(q-last_q)/(now-last_time)
                    if max(velocity)>15:raise RuntimeError('Low-setting jog measured motion exceeds the qualification guard')
                sample={'time':now-began,'q':q.tolist(),'phase':phase,'jog_sdk_settings':settings[:],'moving':c.moving,
                    'position_mm':(ARM.fk(np.deg2rad(q))*1000).tolist(),'error_mm':float(np.linalg.norm(ARM.fk(np.deg2rad(q))*1000-plan['target_mm']))}
                owner.status['samples'].append(sample);last_time=now;last_q=q.copy()
                diagnostic=plan['result']['diagnostic']
                if phase=='waiting':
                    if c.moving:raise RuntimeError('Unplanned motion before jog qualification')
                    c._health()
                    j=JOINTS[0];direction=diagnostic['directions'][0]
                    c._send_motion(lambda:c.robot.jog_angle(j,direction,SDK_SPEED,_async=True))
                    started=True;phase='first_joint';phase_at=time.monotonic();settings[j-1]=(1 if direction else -1)*SDK_SPEED
                    events.append({'time_s':phase_at-began,'command':'jog_angle','joint':j,'direction':direction,'speed':SDK_SPEED})
                    c._deadline(2.5,'Jog qualification watchdog expired; Stop sent')
                elif phase=='first_joint' and now-phase_at>=PHASE_SECONDS:
                    c._health();j=JOINTS[1];direction=diagnostic['directions'][1]
                    if time.monotonic()-began-events[0]['time_s']>=2.5:
                        raise RuntimeError('Jog deadline elapsed during health checks; no further jog command allowed')
                    c._send_motion(lambda:c.robot.jog_angle(j,direction,SDK_SPEED,_async=True))
                    phase='second_joint';phase_at=time.monotonic();settings[j-1]=(1 if direction else -1)*SDK_SPEED
                    events.append({'time_s':phase_at-began,'command':'jog_angle','joint':j,'direction':direction,'speed':SDK_SPEED})
                elif phase=='second_joint' and now-phase_at>=PHASE_SECONDS:
                    c._send_motion(lambda:c.robot._mesg(ProtocolCode.STOP,_async=True))
                    stopped=True;settings=[0]*6;phase='settling';phase_at=time.monotonic()
                    events.append({'time_s':phase_at-began,'command':'stop'})
                    if c.timer:c.timer.cancel();c.timer=None
                elif phase=='settling' and now-phase_at>=1.:
                    if c.moving:raise RuntimeError('Jog did not stop after its Stop command')
                    outcome=analyze(owner.status['samples'],events,start)
                    owner.release_mouse_mode();c.motion=c.target=None
                    owner.status.update(state='completed',message='Directional jog qualification completed',
                        qualification=outcome,jog_events=events)
                    return
                owner.status.update(jog_events=events[:],jog_sdk_settings=settings[:],message=f'Directional jog qualification: {phase}')
    except Exception as exc:
        with c.lock:
            if c.generation!=generation or c.motion!='ik':return
            try:c._stop()
            except Exception:c._close()
            c.error=f'Jog qualification stopped: {exc}'
            owner.status.update(state='failed',message=c.error,jog_events=events)
    finally:
        if scene is not None:scene.close()
        # Cancellation paths already send Stop through the shared owner. A
        # worker shutdown must also clear a jog it started, never another owner.
        with c.lock:
            if started and not stopped and c.generation==generation and c.motion=='ik' and c.robot:
                try:c._stop()
                except Exception:c._close()
