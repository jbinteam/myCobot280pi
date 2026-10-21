#!/usr/bin/env python3
"""Compare one target with small SDK targets through the shared WebUI owner.

Planning is read-only by default. --execute explicitly starts physical trials.
The existing UI Stop/Escape, collision checks, heartbeat and timeout remain active.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import numpy as np


class Comparison:
    def __init__(self,output,joint,speed):
        self.base='http://127.0.0.1:8000/api/';self.token=None;self.execution=None
        self.output=Path(output);self.joint=joint;self.speed=speed
        self.report={'created_at':datetime.now(timezone.utc).isoformat(),'joint':joint,'sdk_speed':speed,
            'scope':'Single-target versus small-waypoint comparison. Same SDK setting and absolute endpoint; '
                    'actual starts may differ by up to 0.5 degrees after the return. Different target cadence '
                    'and reference duration. Not an isolated speed calibration.',
            'trials':[]}
    def request(self,route,body=None):
        req=urllib.request.Request(self.base+route)
        if body is not None:
            req=urllib.request.Request(self.base+route,data=json.dumps(body,allow_nan=False).encode(),
                headers={'Content-Type':'application/json','X-Robot-Token':self.token})
        try:
            with urllib.request.urlopen(req,timeout=15) as r:data=json.load(r)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(json.load(exc).get('error',str(exc))) from exc
        if route=='state':self.token=data['token']
        return data
    def save(self):
        self.output.parent.mkdir(parents=True,exist_ok=True)
        self.output.write_text(json.dumps(self.report,indent=2,allow_nan=False)+'\n')
    def stop_owned(self):
        s=self.request('state')
        if self.execution and s.get('motion')=='ik' and s.get('ik',{}).get('execution_id')==self.execution:
            self.request('stop',{})
    def summarize(self,ik):
        a=ik['samples'];q=np.array([x['q'][self.joint-1] for x in a]);t=np.array([x['time'] for x in a])
        d=np.diff(q);active=d[np.abs(d)>.1]
        sends=[x for x in a if 'sent_command_deg' in x]
        times=np.array([x['command_send_time_s'] for x in sends])
        return {'state':ik['state'],'message':ik['message'],'sample_count':len(a),'sent_targets':len(sends),
            'observed_duration_s':float(t[-1]-t[0]),'measured_net_travel_deg':float(q[-1]-q[0]),
            'measured_absolute_travel_deg':float(np.abs(d).sum()),
            'excess_absolute_travel_deg':float(np.abs(d).sum()-abs(q[-1]-q[0])),
            'reported_reversals_over_0_1_deg':int(np.sum(active[:-1]*active[1:]<0)),
            'unchanged_angle_interval_fraction':float(np.mean(np.abs(d)<.005)),
            'median_send_interval_s':float(np.median(np.diff(times))) if len(times)>1 else None,
            'final_joint_deg':float(q[-1])}
    def trial(self,name,target,variant):
        state=self.request('state')
        if not state['connected'] or state['moving'] or state['motion'] or state['ik'].get('mouse_aim_session_id'):
            raise RuntimeError('Trial needs a connected, idle arm with mouse tracking stopped')
        plan=self.request('ik/joint-diagnostic/plan',{'joint':self.joint,'target_deg':target,
            'variant':variant,'speed':self.speed})
        trial={'name':name,'plan':plan['result']['diagnostic'],'execution':None}
        self.report['trials'].append(trial);self.save()
        self.execution=plan['plan']['id']
        self.request('ik/execute',{'plan_id':self.execution,'speed':self.speed,'armed':True,
            'frame_verified':True,'scene_verified':True})
        self.request('ik/heartbeat',{'execution_id':self.execution})
        deadline=time.monotonic()+max(15,len(plan['result']['frames'])*.6+10)
        while time.monotonic()<deadline:
            state=self.request('state');ik=state['ik']
            if ik.get('execution_id')!=self.execution:raise RuntimeError('Another owner changed the execution')
            trial['execution']=ik;self.save()
            if ik['state']!='running':
                trial['summary']=self.summarize(ik);self.save();self.execution=None
                print(name, json.dumps(trial['summary']),flush=True)
                if ik['state']!='completed':raise RuntimeError('Trial stopped; results saved, no automatic retry')
                time.sleep(1.)
                return
            try:self.request('ik/heartbeat',{'execution_id':self.execution})
            except RuntimeError:
                # Completion can race the last lease renewal. Observe the
                # terminal state rather than discarding the finished trace.
                observed=self.request('state')['ik']
                if observed.get('execution_id')!=self.execution or observed['state']=='running':raise
            time.sleep(.2)
        raise RuntimeError('Comparison deadline exceeded')
    def run(self,amplitude,execute):
        state=self.request('state')
        if not state['hardware_enabled'] or not state['connected'] or state['moving'] or state['motion']:
            raise RuntimeError('Select hardware and connect an idle arm before preparing the comparison')
        start=state['angles'];self.report['baseline_deg']=start[:]
        target=round(start[self.joint-1]+amplitude,2)
        for variant in ('single_target','small_waypoints'):
            p=self.request('ik/joint-diagnostic/plan',{'joint':self.joint,'target_deg':target,'variant':variant,'speed':self.speed})
            self.report.setdefault('prepared_comparisons',[]).append(p['result']['diagnostic'])
        self.save();print('Both comparison paths checked; no movement during preparation.',flush=True)
        if not execute:return
        for name,angle,variant in [('single_target',target,'single_target'),
            ('return_after_single',start[self.joint-1],'single_target'),
            ('small_waypoints',target,'small_waypoints'),
            ('return_after_waypoints',start[self.joint-1],'single_target')]:
            state=self.request('state')
            others=[abs(a-b) for j,(a,b) in enumerate(zip(state['angles'],start)) if j!=self.joint-1]
            if max(others)>.5:raise RuntimeError('An untested joint drifted by more than 0.5 degrees; comparison stopped')
            if name=='small_waypoints' and abs(state['angles'][self.joint-1]-start[self.joint-1])>.5:
                raise RuntimeError('Return did not establish a comparable start')
            self.trial(name,angle,variant)
        self.report['final_angles_deg']=self.request('state')['angles'];self.save()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--joint',type=int,default=4)
    parser.add_argument('--speed',type=int,default=70)
    parser.add_argument('--amplitude',type=float,default=2.)
    parser.add_argument('--output',default='output/joint_command_comparison.json')
    args=parser.parse_args();comparison=Comparison(args.output,args.joint,args.speed)
    try:comparison.run(args.amplitude,args.execute)
    except BaseException as exc:
        try:comparison.stop_owned()
        except Exception as stop_error:comparison.report['stop_error']=str(stop_error)
        comparison.report['error']=str(exc);comparison.save();raise


if __name__=='__main__':main()
