"""Owned-process supervisor for the complete direct ROS arm system."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/ros_system'
RECORD = OUT/'run.json'
sys.path.insert(0, str(ROOT/'object_tracking_ros2'))
from object_tracking_ros2.runtime_options import OPTIONS, validate, controller_overrides


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(record):
    return bool(record.get('start_ticks')) and identity(record['pid']) == record['start_ticks']


def load_record():
    try:return json.loads(RECORD.read_text())
    except (FileNotFoundError, ValueError):return {}


def stop_existing():
    record = load_record()
    if not alive(record):
        survivors=[child for child in record.get('children',[]) if alive(child)]
        for child in survivors:
            os.killpg(child['pid'],signal.SIGINT)
        deadline=time.monotonic()+20
        while any(alive(child) for child in survivors) and time.monotonic()<deadline:time.sleep(.1)
        if any(alive(child) for child in survivors):raise RuntimeError('An owned child is still shutting down')
        print('Stopped remaining owned ROS processes.' if survivors else 'No managed ROS system is running.',flush=True)
        return
    command = Path(f'/proc/{record["pid"]}/cmdline').read_bytes()
    if str(Path(__file__)).encode() not in command:
        raise RuntimeError('Saved PID does not belong to this launcher')
    os.kill(record['pid'],signal.SIGINT)
    deadline = time.monotonic()+25
    while alive(record) and time.monotonic()<deadline:time.sleep(.1)
    if alive(record):raise RuntimeError('System is still shutting down; see '+str(OUT/'latest'))
    print('ROS system stopped.',flush=True)


def service(node, name, seconds=6):
    import rclpy
    from std_srvs.srv import Trigger
    client=node.create_client(Trigger,name)
    try:
        if not client.wait_for_service(timeout_sec=2):raise RuntimeError(name+' is unavailable')
        future=client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node,future,timeout_sec=seconds)
        if not future.done():raise RuntimeError(name+' timed out')
        result=future.result()
        if not result.success:raise RuntimeError(result.message)
        return result.message
    finally:node.destroy_client(client)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('start','stop','arm','launch'))
    p.add_argument('--restart',action='store_true',help='Stop the previous managed run, then start fresh')
    p.add_argument('--no-arm',action='store_true',help='Connect and display; enable tracking later with arm_tracking.sh')
    p.add_argument('--no-rviz',action='store_true',help='Run without opening a desktop window')
    p.add_argument('--no-detector',action='store_true',help='Use detections from an existing ROS publisher')
    p.add_argument('--print-config',action='store_true',help='Print configuration without starting processes or connecting hardware')
    for key, (default, description) in OPTIONS.items():
        p.add_argument('--'+key.replace('_','-'),default=default,help=description)
    p.add_argument('--scanner-root',type=Path,default=ROOT.parent/'objectScanner')
    p.add_argument('--dataset',default='my_object_20260925_r2')
    p.add_argument('--category',default='my_object')
    p.add_argument('--robot-host',default='192.168.77.225')
    p.add_argument('--robot-port',type=int,default=9000)
    return p


def run(args):
    if args.command=='stop':
        stop_existing();return
    settings = validate({key:getattr(args,key) for key in OPTIONS})
    overrides = controller_overrides(settings)
    if not 1 <= args.robot_port <= 65535:
        raise ValueError('robot_port must be between 1 and 65535')
    if args.command=='launch':
        pairs = {**{key:str(value).lower() if isinstance(value,bool) else value for key,value in settings.items()},'project_root':str(ROOT),'scanner_root':str(args.scanner_root),
            'dataset':args.dataset,'category':args.category,'robot_host':args.robot_host,'robot_port':args.robot_port,
            'auto_arm':str(not args.no_arm).lower(),'rviz':str(not args.no_rviz).lower(),
            'detector':str(not args.no_detector).lower(),'restart':str(args.restart).lower(),
            'print_config':str(args.print_config).lower()}
        os.execvp('ros2',['ros2','launch','object_tracking_ros2','full_system.launch.py',
                          *[f'{key}:={value}' for key,value in pairs.items()]])
    if args.print_config:
        print(json.dumps({'settings':settings,'controller_parameters':overrides,
            'robot_host':args.robot_host,'robot_port':args.robot_port,
            'auto_arm':not args.no_arm,'rviz':not args.no_rviz,'detector':not args.no_detector,
            'scanner_root':str(args.scanner_root),'dataset':args.dataset,'category':args.category},indent=2))
        return
    import rclpy
    from std_msgs.msg import String
    rclpy.init()
    node=rclpy.create_node('arm_system_launcher')
    if args.command=='arm':
        try:print(service(node,'/object_follow/arm'),flush=True)
        finally:node.destroy_node();rclpy.shutdown()
        return
    if args.restart:stop_existing()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'launcher.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('System already running. Use --restart or ./stop_ros_system.sh.')
    detector=args.scanner_root/'scripts/collect_process/07_run_live.sh'
    required_files = [ROOT/'.venv/bin/python']
    if not args.no_detector:required_files += [detector,args.scanner_root/'.venv-ros-cuda/bin/python']
    if not args.no_rviz:required_files.append(Path('/opt/ros/jazzy/lib/rviz2/rviz2'))
    for required in required_files:
        if not required.is_file():raise RuntimeError('Missing dependency: '+str(required))
    if not args.no_rviz and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        raise RuntimeError('No desktop display. Run from the desktop terminal, or add --no-rviz.')
    # Do not create a second camera owner or competing controller. A fresh
    # managed run can restart its own processes; external launches stay owned
    # by their original terminal.
    discovery=time.monotonic()+1.5
    while time.monotonic()<discovery:rclpy.spin_once(node,timeout_sec=.1)
    if node.count_publishers('/object_follow/status') or (not args.no_detector and node.count_publishers('/camera/color/image_raw')):
        raise RuntimeError('An external arm/detector launch is already running. Stop its terminal first, then rerun this launcher.')
    stamp=time.strftime('%Y%m%d-%H%M%S')
    logs=OUT/stamp;logs.mkdir()
    latest=OUT/'latest'
    if latest.is_symlink():latest.unlink()
    if not latest.exists():latest.symlink_to(logs.name,target_is_directory=True)
    children=[];halt=False;status={};received=0.;armed_once=args.no_arm;stable=None;arm_future=None
    record={'pid':os.getpid(),'start_ticks':identity(os.getpid()),'children':[],
        'logs':str(logs),'robot_host':args.robot_host,'auto_arm':not args.no_arm,'started':stamp,
        'settings':settings,'controller_parameters':overrides}
    def save():
        temporary=RECORD.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2));temporary.replace(RECORD)
    def spawn(name,command,env=None):
        handle=(logs/(name+'.log')).open('ab')
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
            stdout=handle,stderr=handle,start_new_session=True)
        handle.close();children.append((name,child))
        record['children'].append({'name':name,'pid':child.pid,'start_ticks':identity(child.pid)})
        save()
        print(f'Started {name}; log: {logs/(name+".log")}',flush=True)
    def request_stop(*_):
        nonlocal halt
        halt=True
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    def update(message):
        nonlocal received
        status.clear();status.update(json.loads(message.data));received=time.monotonic()
    node.create_subscription(String,'/object_follow/status',update,10)
    from std_srvs.srv import Trigger
    arm_client=node.create_client(Trigger,'/object_follow/arm')
    try:
        save()
        if not args.no_detector:
            spawn('detector',['bash',str(detector),args.dataset,args.category,
                f'camera_backend:={settings["camera_backend"]}',f'camera_rate:={settings["camera_rate"]}','rviz:=false'])
        spawn('controller',['bash',str(ROOT/'start_object_tracking.sh'),'--hardware',
            f'controller_language:={settings["controller_language"]}',
            f'robot_host:={args.robot_host}',f'robot_port:={args.robot_port}',
            *[f'{key}:={str(value).lower() if isinstance(value,bool) else value}' for key,value in overrides.items()]])
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        env['PYTHONPATH']=str(ROOT/'object_tracking_ros2')+':'+str(ROOT)+':'+env.get('PYTHONPATH','')
        spawn('visualization',[str(ROOT/'.venv/bin/python'),'-m','object_tracking_ros2.visualization'],env)
        if not args.no_rviz:
            spawn('rviz',['/opt/ros/jazzy/lib/rviz2/rviz2','-d',str(ROOT/'object_tracking_ros2/config/full_system.rviz')])
        print('Full ROS system started. Ctrl+C or ./stop_ros_system.sh stops it.',flush=True)
        print('Motion stays disabled.' if args.no_arm else 'Waiting for fresh arm feedback; tracking will arm once and idle until an object is detected.',flush=True)
        began=time.monotonic();last_state=None;ready_printed=False
        while not halt:
            rclpy.spin_once(node,timeout_sec=.1)
            for name,child in children:
                if child.poll() is not None:raise RuntimeError(f'{name} exited ({child.returncode}); see {logs/(name+".log")}')
            now=time.monotonic()
            if received and now-received>5:raise RuntimeError('Arm controller stopped publishing status')
            feedback=status.get('feedback_age_s')
            ready=(status.get('backend')=='direct' and status.get('api_connected')
                and feedback is not None and feedback<.65 and not status.get('feedback_error'))
            if ready and not ready_printed:
                print('Arm connected; measured feedback is healthy. RViz displays live output.',flush=True);ready_printed=True
            if not ready_printed and now-began>30:
                raise RuntimeError('No fresh arm feedback. Check power/network and run start_pi_bridge.sh on the Pi. '+status.get('feedback_error',''))
            good=ready
            stable=(now if stable is None else stable) if good else None
            if not armed_once and stable is not None and now-stable>=1 and arm_client.service_is_ready():
                armed_once=True;arm_future=arm_client.call_async(Trigger.Request())
            if arm_future is not None and arm_future.done():
                result=arm_future.result();arm_future=None
                print((result.message if result.success else 'Arming did not complete: '+result.message+'. Use ./arm_tracking.sh when ready.'),flush=True)
            phase=(status.get('armed'),status.get('paused'),status.get('detector_tracking'))
            if status and phase!=last_state:
                print('Status: '+('ARMED / WAITING' if phase[0] and phase[1] else 'ARMED / TRACK' if phase[0] else 'DISARMED')
                    +' | detector '+('TRACKING' if phase[2] else 'LOST'),flush=True);last_state=phase
    finally:
        signal.signal(signal.SIGINT,signal.SIG_IGN)
        print('Stopping motion and owned ROS processes...',flush=True)
        try:print(service(node,'/arm/stop'),flush=True)
        except Exception as exc:print('Stop service: '+str(exc),flush=True)
        # Signal the whole child group, including ROS launch descendants.
        for name,child in reversed(children):
            try:os.killpg(child.pid,signal.SIGINT)
            except ProcessLookupError:pass
        deadline=time.monotonic()+12
        for name,child in children:
            try:child.wait(timeout=max(.1,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                # SIGTERM still runs the controller's shutdown handlers.
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:child.wait(timeout=3)
                except subprocess.TimeoutExpired:print(f'{name} still exiting; PID {child.pid}',flush=True)
        record['stopped']=time.strftime('%Y%m%d-%H%M%S');save()
        node.destroy_node();rclpy.try_shutdown();lock.close()


if __name__=='__main__':
    try:run(parser().parse_args())
    except Exception as exc:
        print('ROS system: '+str(exc),file=sys.stderr,flush=True);sys.exit(1)
