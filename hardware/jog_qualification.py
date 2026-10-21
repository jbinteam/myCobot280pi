#!/usr/bin/env python3
"""Check bounded two-joint directional jogging through the WebUI owner.

Default: collision-check the test without movement. --execute: J4 at SDK 1,
then J6 at SDK 1, then global Stop. No automatic retry or return movement.
"""
import argparse
import time
from joint_command_comparison import Comparison


class JogQualification(Comparison):
    def run(self,execute):
        state=self.request('state')
        if not state['hardware_enabled'] or not state['connected'] or state['moving'] or state['motion'] or state['ik'].get('mouse_aim_session_id'):
            raise RuntimeError('Qualification needs hardware enabled and a connected idle arm; stop mouse tracking first')
        self.report.pop('joint',None)
        self.report.update(scope='Bounded J4/J6 SDK-1 directional jog qualification; not six-axis velocity calibration',
            baseline_deg=state['angles'])
        p=self.request('ik/jog-qualification/plan',{})
        self.report['plan']=p;self.save()
        print('Jog qualification prepared; no motion sent.',flush=True)
        if not execute:return
        self.execution=p['plan']['id']
        self.request('ik/execute',{'plan_id':self.execution,'speed':1,'armed':True,'frame_verified':True,'scene_verified':True})
        self.request('ik/heartbeat',{'execution_id':self.execution})
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            state=self.request('state');s=state['ik']
            if s.get('execution_id')!=self.execution:raise RuntimeError('Execution ownership changed')
            self.report['execution']=s;self.save()
            if s['state']!='running':
                self.report['final_angles_deg']=state['angles'];self.save();self.execution=None
                print(s['state'],s.get('qualification',s['message']),flush=True)
                if s['state']!='completed':raise RuntimeError('Qualification stopped; no automatic retry')
                return
            try:self.request('ik/heartbeat',{'execution_id':self.execution})
            except RuntimeError:
                observed=self.request('state')['ik']
                if observed.get('execution_id')!=self.execution or observed['state']=='running':raise
            time.sleep(.15)
        raise RuntimeError('Qualification deadline exceeded')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--output',default='output/jog_qualification.json')
    args=parser.parse_args();probe=JogQualification(args.output,4,1)
    try:probe.run(args.execute)
    except BaseException as exc:
        try:probe.stop_owned()
        except Exception as stop_error:probe.report['stop_error']=str(stop_error)
        probe.report['error']=str(exc);probe.save();raise


if __name__=='__main__':main()
