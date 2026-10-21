"""Plot saved candidate-budget results with system Matplotlib; no simulator."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(output):
    rows = [json.loads(line) for line in (output/'results.jsonl').read_text().splitlines()]
    fig, axes = plt.subplots(1,3,figsize=(14,4.5))
    for ax,key,title in zip(axes,('first_solution_seconds','planning_seconds','joint_travel_rad'),
                            ('First verified route (s)','Total planning time (s)','Joint travel (rad)')):
        maximum = .1
        for split,marker,label in (('known','o','Known cases'),('new_controls','^','New controls')):
            group = [r for r in rows if r['split']==split and all(p['success'] for p in r['policies'].values())]
            for mode,color,name in (('streaming','#c1c5cb','Original streaming'),('fair','#167d8d','Candidate limits')):
                x=[r['policies']['eager'][key] for r in group]
                y=[r['policies'][mode][key] for r in group]
                if x:
                    ax.scatter(x,y,c=color,marker=marker,s=32,alpha=.75,label=f'{name}: {label}')
                    maximum=max(maximum,max(x+y))
        maximum*=1.08
        ax.plot([0,maximum],[0,maximum],'--',color='#888',linewidth=1)
        ax.set(xlim=(0,maximum),ylim=(0,maximum),xlabel='Full IK pool + bounded selection',
               ylabel='Streaming policy',title=title)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7,loc='upper left')
    fig.suptitle(f'Per-candidate limits: {len(rows)} trials, synthetic visibility, offline only')
    fig.tight_layout();fig.savefig(output/'comparison.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    main(Path(sys.argv[1]))
