"""Compare clipped and bounded DLS on fixed, independently seeded pose sets."""
import json
import numpy as np
from .kinematics import ARM
from .pose5 import cases, solve
from .solvers import ARTIFACTS


def benchmark():
    suites=[]
    for seed,count in [(9001,24),(20260930,96)]:
        fixtures=cases(seed,count);rows=[]
        for method in ('dls','bounded_dls'):
            results=[solve(np.deg2rad(c['start_deg']),np.array(c['target_mm'])/1000,
                           c['orientation_xy_deg'],method) for c in fixtures]
            row={'method':method,'success':sum(r['converged'] for r in results),'total':count,
                 'mean_error_mm':float(np.mean([r['final_error_mm'] for r in results])),
                 'mean_orientation_error_deg':float(np.mean([r['final_orientation_error_deg'] for r in results])),
                 'mean_iterations':float(np.mean([r['iterations'] for r in results])),
                 'mean_elapsed_ms':float(np.mean([r['elapsed_ms'] for r in results])),
                 'mean_travel_deg':float(np.mean([r['joint_travel_deg'] for r in results])),
                 'outcomes':[{k:r[k] for k in ('converged','final_error_mm','final_orientation_error_deg','iterations','backtracks')} for r in results]}
            rows.append(row)
            print(f'{seed}: {method} {row["success"]}/{count}, XYZ {row["mean_error_mm"]:.2f} mm, direction {row["mean_orientation_error_deg"]:.2f} deg, CPU {row["mean_elapsed_ms"]:.2f} ms',flush=True)
        suites.append({'seed':seed,'count':count,'rows':rows})
    report={'suites':suites,'steps':240,'dt':.05,'speed_rad_s':1.2,'tolerance_mm':1.,
            'orientation_tolerance_deg':1.,'damping':.04,
            'limits_rad':{'min':ARM.lower.tolist(),'max':ARM.upper.tolist()},'joint_limit_profile':ARM.limit_profile,
            'limitations':'Synthetic reachable targets, one arm geometry, no collision checking or physical validation. Original set used in development; fresh set is a single-seed holdout.'}
    (ARTIFACTS/'benchmark_pose5_bounded.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':benchmark()
