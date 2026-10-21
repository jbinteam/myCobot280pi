"""Replay exact cursor snapshots against self-checked 5D IK and joint planning.

Baseline counts are IK goal acceptance, not baseline planning successes. This
module never imports hardware transport or executes robot motion.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from . import service, solvers, pose5
from .kinematics import ARM
from .prm import planning_space


def benchmark(cases_path, output):
    fixtures = json.loads(Path(cases_path).read_text())
    road = service.load_roadmap(service.ARTIFACTS/'roadmap.json', planning_space('self', 1.))
    rows = []
    for case in fixtures['cases']:
        body = {key: case[key] for key in ('start_deg', 'cursor_mm')}
        began = perf_counter()
        result = service.mouse_aim_plan(body, road)
        row = {**case, **{key: result.get(key) for key in ('status', 'converged', 'ik_outcome', 'planning_outcome',
                'final_error_mm', 'final_orientation_error_deg', 'cursor_acceptance', 'ik_ms', 'planning_ms')}}
        row.update(accepted_goal_count=len(result.get('ik', {}).get('goals', [])),
                   diagnostic=result.get('ik', {}).get('diagnostic'),
                   goal_sources=[g['source'] for g in result.get('ik', {}).get('goals', [])],
                   refinement_methods=sorted(set(a['refinement_method'] for a in result.get('ik', {}).get('attempts', []) if 'refinement_method' in a)),
                   elapsed_ms=(perf_counter()-began)*1000)
        if result.get('converged'):
            assert result['validation']=='joint_limits_and_self_collision'
            assert result['cursor_acceptance']['accepted']
        elif not row['accepted_goal_count']:
            row['xyz_only_comparisons']={}
            for method in ('dls','seed_dls'):
                comparison=solvers.solve(np.deg2rad(body['start_deg']),np.array(result['target_mm'])/1000,method=method)
                end=np.deg2rad(comparison['frames'][-1]['q'])
                row['xyz_only_comparisons'][method]={'converged':comparison['converged'],'collision_valid':road.space.valid(end),
                    'position_error_mm':comparison['final_error_mm'],
                    'target_direction_error_deg':pose5.axis_error(ARM.transform(end)[:3,2],result['target_axis'])[2]}
        rows.append(row)
        print(f"{case['id']}: {result['status']}; {row['accepted_goal_count']} goals; {row['elapsed_ms']:.0f} ms", flush=True)
    report = {'simulation_only': True, 'cases': str(cases_path), 'total': len(rows),
              'previous_tasks_with_accepted_ik_goals':sum(r['previous_accepted_goal_count']>0 for r in rows),
              'tasks_with_accepted_ik_goals':sum(r['accepted_goal_count']>0 for r in rows),
              'validated_plans':sum(bool(r['converged']) for r in rows),
              'recovered_tasks':[r['id'] for r in rows if r['accepted_goal_count'] and not r['previous_accepted_goal_count']],
              'lost_ik_tasks':[r['id'] for r in rows if not r['accepted_goal_count'] and r['previous_accepted_goal_count']],
              'xyz_only_valid_success_among_5d_failures':sum(any(c['converged'] and c['collision_valid'] for c in r.get('xyz_only_comparisons',{}).values()) for r in rows),
              'mean_ik_ms':float(np.mean([r['ik_ms'] for r in rows])), 'rows':rows,
              'limitations':'75 cursor snapshots from three valid starts/planes; not a workspace-wide reachability proof. '
                            'Old baseline records IK goal acceptance only. Discretized self-collision checks exclude external obstacles/tools/cables.'}
    Path(output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}, indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=Path,default=Path(__file__).parent/'fixtures/mouse_aim_recovery_cases.json')
    parser.add_argument('--output',type=Path,default=Path('output/mouse_aim_recovery_benchmark.json'))
    args=parser.parse_args();benchmark(args.cases,args.output)
