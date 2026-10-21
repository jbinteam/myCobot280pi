"""Plot saved streaming IK timing/quality comparisons using system Matplotlib."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(output):
    rows = [json.loads(line) for line in (output/'results.jsonl').read_text().splitlines()]
    paired = [r['policies'] for r in rows if r['policies']['eager_bounded']['success'] and r['policies']['streaming_bounded']['success']]
    fig, axes = plt.subplots(1, 3, figsize=(13.5,4.2))
    for ax,key,title in zip(axes, ('first_solution_seconds','planning_seconds','joint_travel_rad'),
                            ('Time to first visible route (s)','Total planning time (s)','Joint travel (rad)')):
        x = [r['eager_bounded'][key] for r in paired]; y = [r['streaming_bounded'][key] for r in paired]
        ax.scatter(x,y,color='#187c83',s=35,alpha=.8)
        maximum = max(x+y+[.1])*1.08
        ax.plot([0,maximum],[0,maximum],'--',color='#999999')
        ax.set(xlim=(0,maximum),ylim=(0,maximum),xlabel='Full IK pool + bounded search',ylabel='On-demand IK + bounded search',title=title)
        ax.grid(alpha=.2)
    fig.suptitle(f'Streaming IK: {len(paired)} paired visible successes')
    fig.tight_layout(); fig.savefig(output/'comparison.png',dpi=160); plt.close(fig)


if __name__=='__main__':
    main(Path(sys.argv[1]) if len(sys.argv)>1 else Path('output/streaming_ik_20261012'))
