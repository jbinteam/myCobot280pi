"""Render checked start/middle/goal reference poses from a held-out rollout."""
import argparse
import json
from pathlib import Path
import numpy as np
from .common import ARTIFACTS, write_json
from .dataset import read_episodes
from .env import ArmEnv


def preview(dataset,report,output,episode=None):
    report=json.loads(Path(report).read_text());_,episodes=read_episodes(dataset,'test')
    rows=report['rows']
    row=next((r for r in rows if r['episode']==episode),None) if episode else next(r for r in rows if r['success'] and r['direct_blocked'])
    if row is None:raise ValueError('Requested episode not found')
    expert=next(e for e in episodes if e['id']==row['episode']);states=np.asarray(row['trajectory_rad'])
    output=Path(output);images=[]
    with ArmEnv() as env:
        env.set_obstacle(expert['obstacle'])
        if not env.path_valid(states):raise ValueError('Stored rollout fails current collision checks')
        for name,index in (('start',0),('middle',len(states)//2),('goal',len(states)-1)):
            env.reset(states[index]);destination=output/f'{name}.png';env.render(destination);images.append(str(destination))
    result={'episode':row['episode'],'success':row['success'],'direct_blocked':row['direct_blocked'],
            'executed_steps':row['executed_steps'],'images':images}
    write_json(output/'preview.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=ARTIFACTS/'demonstrations.h5')
    parser.add_argument('--report',type=Path,default=Path('output/diffusion/evaluation.json'))
    parser.add_argument('--output',type=Path,default=Path('output/diffusion/preview'))
    parser.add_argument('--episode')
    args=parser.parse_args();print(preview(args.dataset,args.report,args.output,args.episode))


if __name__=='__main__':main()
