"""Plot saved bounded-search results with system Matplotlib; no robot imports."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(output):
    rows = [json.loads(line) for line in (output/'results.jsonl').read_text().splitlines()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for split, color, label in [('previous_pilot', '#197c83', 'Previous scenarios'),
                                 ('held_out_controls', '#db8e22', 'Held-out controls')]:
        paired = [r['policies'] for r in rows if r['split'] == split
                  and r['policies']['bounded_visible']['success'] and r['policies']['best_visible']['success']]
        for ax, key in zip(axes, ('joint_travel_rad', 'planning_seconds')):
            ax.scatter([r['best_visible'][key] for r in paired],
                       [r['bounded_visible'][key] for r in paired], color=color, label=label, s=35, alpha=.8)
    for ax, key, label in zip(axes, ('joint_travel_rad', 'planning_seconds'), ('Joint travel (rad)', 'Planning time including shared IK (s)')):
        values = [r['policies'][mode][key] for r in rows for mode in ('best_visible','bounded_visible')
                  if r['policies'][mode]['success']]
        maximum = max(values+[.1])*1.08
        ax.plot([0,maximum], [0,maximum], '--', color='#999999')
        ax.set(xlim=(0,maximum), ylim=(0,maximum), xlabel='Full search', ylabel='Bounded improvement', title=label)
        ax.grid(alpha=.2)
    axes[0].legend(loc='upper left', fontsize=9)
    fig.suptitle(f'350 ms / 3-attempt improvement budget: {len(rows)} paired trials')
    fig.tight_layout(); fig.savefig(output/'comparison.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    main(Path(sys.argv[1]) if len(sys.argv)>1 else Path('output/endpoint_early_stop_20261012'))
