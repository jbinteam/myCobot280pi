#!/usr/bin/env python3
"""Render the saved command comparison; reads files only, no robot connection."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

r=json.loads(Path('output/joint_command_comparison.json').read_text())
j=r['joint']-1
fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
for ax,name,title in zip(axes,('single_target','small_waypoints'),('One target','Small waypoints')):
    trial=next(x for x in r['trials'] if x['name']==name)
    a=trial['execution']['samples'];events=[x for x in a if 'sent_command_deg' in x]
    t=[x['time'] for x in a];q=[x['q'][j] for x in a]
    commands=[trial['plan']['start_deg'][j],*[x['sent_command_deg'][j] for x in events],events[-1]['sent_command_deg'][j]]
    times=[t[0],*[x['command_send_time_s'] for x in events],t[-1]]
    ax.step(times,commands,where='post',color='#3075ae',label='SDK position target',linewidth=1.6)
    ax.plot(t,q,'o-',markersize=3,color='#bb642f',label='Reported joint angle',linewidth=1.1)
    delta=trial['plan']['end_deg'][j]-trial['plan']['start_deg'][j]
    ax.set_title(f'{title}: {len(events)} packet(s), {delta:.2f}° requested',fontsize=11,pad=12)
    ax.set_xlabel('Elapsed time (s)');ax.grid(alpha=.2);ax.legend(loc='upper left',fontsize=9)
axes[0].set_ylabel(f'J{j+1} angle (degrees)')
fig.suptitle(f'J{j+1} command granularity comparison — SDK speed {r["sdk_speed"]}',fontsize=14)
fig.text(.5,.015,'Reported angles are sampled at about 0.30 s; telemetry cannot establish continuous physical motion.\n'
    'Trial starts differ after the return. This comparison does not calibrate a minimum reliable speed.',ha='center',fontsize=9,color='#444444')
fig.tight_layout(rect=(0,.105,1,.92))
fig.savefig('output/joint_command_comparison.png',dpi=170)
print('Saved output/joint_command_comparison.png')
