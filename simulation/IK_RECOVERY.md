# 5D seed coverage and solver recovery

Mouse aim and the WebUI's 5D PRM button enable this recovery automatically.
The original XYZ mode and its checkpoints are retained. Other Python /
`/api/sim/prm` 5D callers can opt in
with `pose5_global_recovery=True`; classical standalone controller experiments
continue to measure their selected controller without this planning recovery.
This code never imports robot transport or prepares hardware motion.

## Search policy

The normal stage retains all four raw NN checks, independent finite/in-limit
NN refinement eligibility, raw-goal-dependent conventional fallback, and unique
seed order **NN → current → up to four roadmap → four deterministic random**.
The default accepted-goal budget is four. Initially colliding optimizer seeds
are permitted; executed/previewed paths must pass the existing validity checks.

The selected numerical controller runs first on each seed; requested classical
recovery methods and automatic bounded DLS retain their existing order. The
normal seed queue and goal budget still determine termination. Trust-region
optimization is reserved for the separate recovery pass, so normal successful
queries do not pay for another solver on each unsuccessful seed.

If this stage produces no valid goal, a separate coverage stage uses at most
32 unique seeds, in this order:

1. Up to four promising numerical endpoints for continuation.
2. Original NN guesses and the current configuration.
3. Up to eight diverse configurations from the entire compatible roadmap.
4. Four trained NN proposals around each of two other bounded configuration
   anchors. These anchors are optimizer inputs, not readings of the current
   simulated or physical robot. The checkpoint's original per-anchor proposal
   bounds and normalization still apply. Missing/incompatible checkpoints skip
   this family and retain numerical recovery.
5. Deterministic Latin-hypercube configurations spanning the operating limits.

Roadmap seeds retain the original pose score:
`norm(XYZ_node - XYZ_target) / 0.3 m + 0.6 * directed_axis_angle_rad`.
Greedy selection also requires separation of at least 0.55 in joint-radius
normalized first-five-joint distance. Joint 6 is excluded from *seed diversity*
because it only spins the flange around local tool Z in this URDF; it remains
free in the optimizer and is checked for limits, collision and motion validity.
Each stratified joint visits each of twelve range bins once. Seed count limits
truncate the queue after removing duplicates. RNG seed is reproducible.

If accepted goals cannot attach/route, one additional coverage pass expands
the goal budget to eight and skips previously tried coverage seeds. New goals
must be at least 0.025 apart in the planner's normalized full joint-range metric.
Existing goals are preserved. Routing is retried only if a new goal was added.
This can include meaningful wrist-spin alternatives without requiring spin.
Final-path and trajectory-validation failures do not restart IK. There is no
unbounded retry loop. Numerical fallback disabled means recovery is disabled.

## Bounded numerical optimizer

[`ik_recovery.py`](ik_recovery.py) uses SciPy's
[trust-region reflective least-squares solver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
with operating joint bounds, an analytic FK Jacobian, and at most 96 optimizer
function evaluations (`max_nfev`) per seed. Analytic derivative/acceptance checks
are additional FK computations. An exactly antipodal start also uses thirteen
bounded escape probes, recorded separately as `escape_probes`.
The dependency is already pinned in
`requirements-collision.txt`; no dependency, trained checkpoint or roadmap is
replaced. Lazy import keeps the XYZ inference dependency unchanged.

The residual is
`[(p_actual - p_target) / ARM.scale, 0.6 * (u_actual - u_target)]`, using meters,
radians, and directed unit tool-Z vectors in **g_base** from **joint6_flange**.
It has six components and rank five. The orientation part contributes
`0.6² * (1 - dot(u_actual, u_target))` to the least-squares cost. Alignment has
a unique direction minimum; opposite directions do not match. No yaw, full
rotation matrix, or local-spin residual is added. An exactly antipodal starting
axis receives a deterministic bounded perturbation to escape its stationary
maximum. Acceptance still uses the robust atan2 angular error.

Optimizer convergence is insufficient for acceptance. Position and direction
tolerances, finite values, operating limits, configured self-collision criteria,
and actual mouse-cursor gap/angle constraints must all pass. A stationary invalid
endpoint remains an IK failure. Optimizer iterations/evaluations are computation;
their states are never used as playback motion. Joint-space direct/PRM planning,
shortcuts, final validation and speed-limited trajectory generation remain
independent. The target-offset convention `base + R_start @ local_offset` is
unchanged, and no IK is repeated at known joint waypoints.

## Diagnostics and reproduction

The response retains normal attempts and seed queue, plus coverage trust-region
method/evaluation/termination information and optional `ik.recovery.phases`.
Each phase records its trigger, seed/evaluation/goal budgets, actual used queue,
added goals and elapsed time. IK and planning time remain separate when routing
recovery occurs. Mouse aim shows recovery goal sources and coverage counts.

Python example: `IKConfig(task='pose5', nn='seed5', pose5_global_recovery=True)`.
The normal goal budget is `max_candidates=4`; `recovery_seeds=32`,
`recovery_evaluations=96`, and `recovery_max_candidates=8` bound the extra search.
Mouse aim fixes these settings alongside its full validation policy.
The planning CLI accepts `--pose5-global-recovery`, `--recovery-seeds`,
`--recovery-evaluations` and `--recovery-max-candidates` for the same policy.

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s simulation/tests -v
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.benchmark_ik_recovery \
  --count 64 --seed 20261004 --output output/ik_recovery/benchmark.json
# For a shorter repeat, select --only fk or --only cursor.
```

The benchmark generates 64 valid full-range/nearby/boundary FK tasks and eight
same-XYZ direction-family tasks, with witnesses retained for audit but never
supplied to IK. It compares single-current numerical IK, the previous full NN
search, and upgraded search/routing. It also replays all 75 existing cursor
snapshots against the old and upgraded pipeline. Valid IK, validated routes,
both goal errors, lost/recovered tasks, latency, and recovery usage are recorded.
Per-task times are one unprofiled run, excluding cold graph/model initialization;
they are not statistically exhaustive latency measurements. Failed search does
not prove unreachability. Existing collision/environment limitations remain.

Workflow: [Mermaid](../output/workflow/mycobot_mouse_aim_full.mmd),
[PNG](../output/workflow/mycobot_mouse_aim_full.png),
[SVG](../output/workflow/mycobot_mouse_aim_full.svg).

## Validation results (2026-10-02)

The final comparison is [benchmark.json](../output/ik_recovery/benchmark.json).

| Sample | Previous full NN search | Recovery enabled |
|---|---:|---:|
| Valid IK and validated routes, 72 FK/witness tasks | 71/72 | 72/72 |
| Valid IK and validated routes, 75 cursor snapshots | 56/75 | 56/75 |
| Lost successful tasks across both samples | — | 0 |
| Median time on the 71 previously successful FK tasks | 355.7 ms | 355.2 ms |
| Median time on the 56 successful cursor snapshots | 141.3 ms | 140.1 ms |

Single-current classical DLS solved/planned 42/72 FK tasks. The additional
recovered task `fk-8` uses a stratified recovery seed; its validated direct path
ends at 0.958 mm position error and 0.293 degrees axis error. It takes about
1.678 s including 0.495 s of coverage search. The 71 previously successful FK
tasks and 56 cursor successes retain their exact accepted-goal counts. No new
checkpoint was trained or selected.

The 19 failed cursor snapshots still fail the stronger search. The median
additional failed-cursor computation is **462 ms**; overall cursor p95 increases
from 1.230 s to 1.785 s. This is a deliberate, bounded robustness cost on failed
searches, not a speedup or proof of unreachability. Normal successful searches
do not run this extra stage. The sampled benchmark plans are all direct; routing
expansion and obstructed PRM paths are covered separately by regression tests.

An earlier per-seed trust-region experiment added latency to normal successful
queries. That policy was not retained; its measurement is archived separately in
[per_seed_experiment.json](../output/ik_recovery/per_seed_experiment.json).

All **178 automated checks passed**: 110 simulation, 45 bridge/API and 23 browser
checks. The production build passed. New coverage includes analytic residual
derivatives, local-Z spin and opposite directions, clipped-DLS recovery,
stationary invalid results, finite/limit/collision acceptance, stratification,
roadmap posture diversity, anchored NN bounds, distinct goal budgets,
fallback gating, routing expansion, seed reuse and trajectory-failure separation.
The browser checks exercise real 5D/XYZ planning and desktop/mobile layouts.

Four [live HTTP checks](../output/ik_recovery/live.json) passed: recovered FK
target, already-at-cursor, nearby cursor and a correctly held failed cursor.
The already/nearby cursor calls do not invoke coverage. The running production
server is hardware-disabled, disconnected and idle. SHA-256 checks confirm the
four original XYZ assets, selected 5D JSON/PT checkpoint, and roadmap are unchanged.

Implementation: [`ik_recovery.py`](ik_recovery.py),
[`planning.py`](planning.py), [`service.py`](service.py),
[`mouse-aim.js`](../webui/src/mouse-aim.js),
[`simulation-controls.js`](../webui/src/simulation-controls.js).
Benchmark: [`benchmark_ik_recovery.py`](benchmark_ik_recovery.py).
Checks: [`test_ik_recovery.py`](tests/test_ik_recovery.py),
[`test_mouse_aim.py`](tests/test_mouse_aim.py),
[`mouse-aim.spec.js`](../webui/tests/mouse-aim.spec.js),
[`seed5.spec.js`](../webui/tests/seed5.spec.js).
Matching documentation is updated in `MOUSE_AIM.md`, `SEED5.md`, `README.md` and
`PERFORMANCE.md`; workflow sources/renders and UI previews are updated in
`output/workflow/mycobot_mouse_aim_full.*` and `webui/preview/mouse-aim-full-*.png`.

## Cursor failure audit (2026-10-02)

The exact 19 failed cursor fixtures were replayed with the production search,
unchanged 1 mm / 1 degree tolerances, 1 mm self-clearance and existing assets.
All 19 failures reproduced. None reached routing, and none had a candidate
rejected solely by actual cursor gap/aim constraints. Full attempts, recovery
endpoints, bound proximity and collision diagnostics are retained in
[`cursor_failure_audit.json`](../output/ik_recovery/cursor_failure_audit.json).
Its best audited valid endpoint minimizes the largest of the four tolerance
ratios; the planner's `diagnostic.best_valid_error` uses its existing sum score.
These two diagnostics can therefore select different endpoints.

A separate conservative geometry audit excludes **17 of the 19 targets**
within the current URDF, operating limits and nominal 1 mm / 1 degree
tolerances. It considers both base-yaw branches and both wrist branches,
checking necessary joint-1/joint-5 intervals and the 14.4--206.4 mm planar
two-link reach. All intervals include task tolerances and uniform model-error
bounds for the URDF's near-right-angle origins (at most 0.002465 mm position
and 0.000017736 unit-axis chord relative to the ideal decomposition).
These exclusions do not depend on solver convergence. They are audit evidence;
the production planner does not use them to skip IK.

The two remaining cases are:

- `start-0-cursor-120-120`: 32 pose-accurate refined endpoints all intersected
  `g_base` with `joint5` in the existing convex collision model. The production
  diagnostic counted 36 accurate invalid attempts including raw checks.
  This establishes rejection of the found solutions, not that every feasible
  posture collides.
- `start-0-cursor--60-120`: the best audited valid endpoint has approximately
  3.834 mm nominal position error and 0.463 degrees direction error, with
  joint 2 at its lower operating bound. The conservative geometric intervals
  remain inconclusive. More seeds are not justified by this audit alone.

[`cursor_geometry_audit.json`](../output/ik_recovery/cursor_geometry_audit.json)
records every branch and retains all 56 saved successful cursor tasks as
controls; none are excluded. Five new audit tests passed, covering 1,012 exact
FK witnesses, 1,000 witnesses at the 1 mm / 1 degree tolerance edges, 500
exact/ideal model comparisons, saved cursor controls and endpoint acceptance.
All seven previously recorded XYZ/checkpoint/roadmap hashes were unchanged.
No training, production solver/policy changes, hardware motion or UI restart
was performed in this audit.

Reproduction (offline simulation):

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.audit_cursor_failures
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.cursor_geometry_audit
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest simulation.tests.test_cursor_failure_audit -v
```

The first command revalidates the existing roadmap before replaying only the
19 failures. The geometry command uses saved fixture inputs without invoking
IK or routing. Current conclusions apply to this simulated model and its
operating limits, not calibrated physical reachability. A focused next
investigation can examine the near-boundary case's achievable error tradeoff
and the collision case's alternate posture family.
