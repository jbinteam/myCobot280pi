"""Render saved endpoint comparisons with a Python that has Matplotlib."""
import json
from pathlib import Path
import sys


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [json.loads(line) for line in (output/'results.jsonl').read_text().splitlines()]
    pairs = [(r['policies'][1], r['policies'][2]) for r in rows
             if r['policies'][1]['success'] and r['policies'][2]['success']]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, label in zip(axes, ('joint_travel_rad', 'planning_seconds'), ('Joint travel (rad)', 'Planning time (s)')):
        x = [a[key] for a,b in pairs]; y = [b[key] for a,b in pairs]
        ax.scatter(x, y, color='#167b80', s=35)
        maximum = max([*x, *y, .1])*1.07
        ax.plot([0, maximum], [0, maximum], '--', color='#999999')
        ax.set(xlim=(0,maximum), ylim=(0,maximum), xlabel='First visible', ylabel='Best visible', title=label)
        ax.grid(alpha=.2)
    fig.suptitle(f'Endpoint selection pilot: {len(pairs)} paired visible successes')
    fig.tight_layout(); fig.savefig(output/'comparison.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    plot(Path(sys.argv[1]))
