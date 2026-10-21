# Reusable joint-space via-point planning

The planner separates **finding an accurate goal configuration** from **finding a
valid joint path to that configuration**. It reuses sampled joint configurations
and their full FK transforms. The geometric workspace for a fixed robot, base,
tool and joint limits does not depend on the starting joints. Via points can help
paths and numerical convergence; they cannot make an unreachable target reachable.

This implementation is simulation/dry-run only. Model inputs and checkpoints are
unchanged. PRM now defaults to self-collision checking; see [SELF_COLLISION.md](SELF_COLLISION.md)
for its geometry assumptions, additional dependencies and updated commands.
Existing XYZ inference and an explicit limit-only ablation still need only NumPy.

## Repository inspection and integration

No applicable `AGENTS.md` was found in the project or ancestor directories.
Existing architecture is documented in `README.md`, `POSE5.md`, and
`../webui/{README,PHASE2}.md`.

| File / interface | Actual behavior |
|---|---|
| `kinematics.py`: `ARM.transform(q)`, `ARM.fk(q)`, `ARM.spatial_jacobian(q)` | Six radians in chain order; full 4×4 transform, XYZ meters, geometric spatial Jacobian. Shared local URDF, `g_base` → `joint6_flange`; no tool offset. |
| `joint_limits.json`, `ARM.lower/upper` | Intersect URDF, SDK and recorded Atom 6.5 bounds, inset 2° and rounded inward. J1 ±165.9°, J2 ±133°, J3 ±147.9°, J4 ±143°, J5 −152.9°…157.9°, J6 ±177.9°. |
| `solvers.py`: `solve(q_start,target,method,steps=240,speed=1.2,tolerance=1.,damping=.04)` | Bounded 0.05 s controller rollouts; XYZ meters in, degree/mm frames out. Numerical DLS is reused for endpoint refinement. Tolerance is mm; speed is joint-vector rad/s. |
| `pose5.py`: `solve(q_start,target,orientation_xy_deg=None,...,target_axis=None)` | XYZ plus directed flange Z; local Z spin free. Reused five-row numerical solver, including bounded DLS. |
| `seed5.py`, `seed5_data.py`, `train_seed5.py`, `artifacts/seed5_v1/` | Separate versioned 12-input / four-proposal 5D NN, FK-labeled valid pairs, best-of-four training and checkpoint compatibility checks. See `SEED5.md`. |
| `train.py`, `artifacts/{models,training}.json`, five `.pt` files | Five two-layer 128-unit tanh MLPs; 24,000 training / 2,048 validation FK samples. Serving uses NumPy weights in `models.json`, not PyTorch. Loader checks URDF and normalization limits. |
| `workspace.py`: `transforms(q_batch)`, `geometric_estimate(body)`, `estimate(body)` | Shared batched URDF FK; sampled and solver-tested clouds. Exported clouds retain Cartesian positions only, so cannot supply joint PRM nodes. |
| `artifacts/dataset.npz` | Retains `q`, `goal`, `target`, validation arrays. `fk_samples(path)` checks adjacent training metadata and reuses the joint arrays; FK poses are recomputed. |
| `service.py`: `run(body)` | Existing target offset uses the full starting flange rotation, including spin. Preserved for the new `plan(body, roadmap=None)`. |
| `../webui/hardware_ik.py`: `HardwareIK.prepare/execute` | Reads live joints; server-owned one-use plans; telemetry-paced `send_angles` commands. PRM dry runs do not create these plans. Shared waypoint conversion was extracted without changing existing execution behavior. |
| Existing tests / failures | `tests/test_{simulation,pose5,workspace}.py`; `solvers.cases(seed=9001,count=24)` and `pose5.cases(...)`; saved benchmarks include neural failures and numerical pose5 stalls. Hardware suites use fake robots. |

All movable joints in this URDF are **bounded revolute**, including J6. Joint
interpolation uses `b-a`, never modulo 2π. A J6 motion from +177° to −177° goes
through zero. This implementation is specific to this installed joint topology;
a robot with continuous/prismatic joints needs a corresponding metric and cache
schema extension.

### Existing NN roles

Let `q_norm=(q-ARM.center)/ARM.radius`, `e=(target-FK(q))/0.3`, with meters inside.
“Current pose” in the learned local controllers means actual **joint angles**, not
just Cartesian XYZ. The caller must supply current joint telemetry or an explicit
simulation start.

| Network | Features | Output / use |
|---|---|---|
| `direct` | `target/0.3` (3) | Absolute `center+radius*tanh(output)` (6). FK-checked proposal, optionally DLS-refined. Intentionally target-only. |
| `seed` | `[target/0.3,q_norm]` (9) | Four bounded `clip(q+radius*tanh(output))` seeds (24). New default: validate/refine several endpoints independently of motion. |
| `conditional` | `[q_norm,e]` (9) | `0.12*tanh(output)` joint increment, with the existing speed cap. Evaluate a rollout, never promote one increment to a goal. |
| `differential` | `[q_norm,e]` (9) | `1.2*tanh(output)` rad/s, integrated at 0.05 s. Same checked rollout adapter. |
| `null` | `q_norm` (6) | Posture velocity projected into the DLS null space by the existing solver. Preserved; not used for PRM edges. |

Existing `seed_dls` remains unchanged: it picks one seed using FK-error/travel cost,
moves toward it under a speed cap, then refines within a shared rollout budget.
The new endpoint generator can refine multiple seeds independently; its IK traces
are computations, not paths to execute. Endpoint validity and FK accuracy are
checked before a candidate is admitted. Numerical refinement may cross invalid
states internally; the path planner separately validates the actual joint motion.
In-limit NN seeds can be refined even if self-colliding; final endpoints still
must be valid. The WebUI's selected 5D classical controller can now supply
numerical endpoints, with bounded DLS fallback per seed. API/CLI defaults remain
bounded DLS; select another method with `pose5_method` / `--pose5-method`.

Local NN rollouts check every transition, a finite budget and best-error progress
(default 30-frame window, 0.01 mm minimum improvement). Their diagnostics survive
fallback to refinement. PRM edges use conventional joint interpolation, independent
of these policies, so the graph is undirected. The known joint controller follows
the resulting configurations directly; no IK is run at each waypoint.

No missing joint-state conditioning was discovered in the local models. Existing
XYZ weights cannot control pose5. The new `seed5` proposal method has its own
version/task identifier, twelve inputs (six normalized joints, three scaled XYZ
coordinates and three directed unit-axis components), FK data and position/axis
losses. It never loads XYZ weights as orientation-capable weights. Missing or
incompatible 5D weights are reported and supported numerical fallback remains
available. The WebUI defaults to `seed5` in 5D and `seed` in XYZ mode.

5D roadmap seeds are ranked by `norm(node_XYZ-target)/0.3 m + 0.6*axis_angle_rad`.
The directed angle uses `atan2(norm(cross), dot)`, so opposite directions have a
π-radian penalty. Goal orientation changes do not invalidate joint connectivity;
the existing robot, geometry, limits, base/TCP and collision metadata still must
match. Orientation is checked at accepted endpoints, not along PRM edges.

## Build, query, inspect and benchmark

Run from `~/codex/projects/myCobot280pi`:

```bash
# Install collision dependencies, then reuse 200 FK samples + 200 uniform samples.
.venv/bin/python -m pip install -r simulation/requirements-collision.txt
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.prm \
  --samples 400 --neighbors 12 --max-step-rad 0.04 --seed 9001 \
  --dataset simulation/artifacts/dataset.npz

# Reuse the saved roadmap for a target query; output includes preview frames.
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.planning \
  --start-deg 0 -40 -130 80 0 50 --target-mm 100 0 200 \
  --output simulation/artifacts/prm_query.json

# Add another 400 nodes to the same cache, preserving existing edges.
# Use the same build options: mismatched configuration is rejected.
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.prm \
  --grow --samples 400 --neighbors 12 --max-step-rad 0.04 --seed 9001 \
  --dataset simulation/artifacts/dataset.npz

# All ablations share the same reproducible 24 start/target cases.
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.benchmark_prm \
  --count 24 --samples 400 --seed 9001 \
  --output simulation/artifacts/benchmark_prm_self.json

OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s simulation/tests -v
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s webui/tests_python -v
```

Omit `--dataset` to sample entirely uniformly. `--no-refine --no-fallback` isolates
raw NN accuracy or local NN rollout behavior; `--nn none` selects numerical IK.
For pose5, explicitly use `--task pose5 --nn none --orientation X Y`. Available
query settings include steps, speed, damping, candidate count, random/roadmap seed
counts, tolerances and RNG seed. `--shortcut` validates every replacement segment.

The new `POST /api/sim/prm` uses the existing local session token. It accepts
`start_deg`, `target_mm`, `task`, `orientation_xy_deg`, `eef_offset_mm`, `nn`,
`refine`, `numerical_fallback`, `pose5_method`, seed counts, `max_candidates`, speed, steps, damping,
tolerances, seed, `shortcut`, `collision` and `self_clearance_mm`. The server loads `artifacts/roadmap.json`; arbitrary
client cache paths are not accepted. While connected and stopped, it captures live
joints and overrides client starts. It returns a dry-run result without a hardware
plan ID. The WebUI's separate PRM button previews these results; `/api/sim/solve` remains unchanged. The new
planner is exposed through Python, CLI and HTTP, with the same viewer frame format.

The UI's **Repair blocked routes** checkbox enables a 32-node guided fallback
for PRM previews and mouse aim. API callers opt in with `guided_route_repair:
true`; `route_repair_samples` accepts an integer from 1 to 128 (default 32).
Repair runs once after normal routing and any existing IK goal expansion fail.
The `guided_repair` response records enabled/triggered/recovered flags, the
original route failure, sample/edge counts, and phase timings. Final trajectory
validation remains required. See [GUIDED_REPAIR.md](GUIDED_REPAIR.md).

### Python interfaces and validity callbacks

```python
from simulation.prm import JointSpace, PRMConfig, Roadmap, planning_space
from simulation.planning import IKConfig, query

# Supply a deterministic state checker with explicit cache identity.
# This example constraint is synthetic; it does not model physical collisions.
space = JointSpace(
    validity=lambda q: (q[0]**2 + q[1]**2) >= 0.35**2,
    assumptions={"checker": "synthetic_disk_v1", "environment": "q1_q2_radius_0.35"},
)
road = Roadmap(space, PRMConfig(samples=400, max_step_rad=0.04)).grow()
road.save("/tmp/mycobot-synthetic-prm.json")
road = Roadmap.load("/tmp/mycobot-synthetic-prm.json", space)
# query(q_current_rad, target_m, road, IKConfig(), orientation_xy_deg=None)
# road.plan(q_current_rad, validated_goal_arrays, shortcut=True) isolates routing.
```

`generate_goals(...)` and `Roadmap.plan(...)` are separate reusable interfaces.
The former returns valid FK-accurate goals, attempted sources/errors and IK time;
the latter requires joint endpoints already validated for the target. Roadmap FK
neighbors can be numerical seeds, but routing cannot repair inaccurate IK.
Default numerical fallback uses the current joints, four pose-near roadmap nodes
and four deterministic uniform seeds. `max_candidates=4` limits accepted branches;
seed/rollout budgets bound the work. There is no complete IK search guarantee.

## Motion validation, cost and persistence

Every node retains six joint radians and its full flange FK transform. Connections
use the nearest `k` nodes by `norm((b-a)/(upper-lower))`; the undirected union of
neighbor lists is retained. Dijkstra uses this same nonnegative edge cost. Path
length is additionally reported as the sum of Euclidean segment norms in radians.

Endpoints and each interpolated state must satisfy limits and the optional
callback, with at most `max_step_rad` change per joint. In the explicit limit-only
case, the entire linear segment is valid analytically because the box is convex.
Callback checks are discretized, not continuous collision detection: narrow
invalid regions can fall between samples. Use an appropriate resolution and a
conservative checker. Final paths, shortcut segments and denser speed-derived
preview intervals are checked again. Frames use the existing 0.05 s joint-vector
speed cap. `trajectory.joint_waypoints` also supplies bounded degree commands
without compressing PRM paths; removing a point would require revalidation.

Cache metadata records schema, robot, URDF and referenced mesh hashes, joint names
and topology, exact limits, radians/meters, base frame/transform, flange/tool offset,
validity mode, environment/checker identity, metric and build configuration.
Self-collision assumptions also identify clearance, excluded pairs, convex
representation and native-library versions. Load
rejects changed assumptions and rechecks nodes, FK metadata, edge validity and
costs. Callback users must update the checker/environment identifiers whenever the
implementation, obstacle transforms, base, tool, clearance or validity rules change.
Query attachments are temporary; the reusable graph is not modified. Growing a
roadmap samples deterministically from the seed and current node count.

Default results now say `joint_limits_and_self_collision`, using conservative
convex hulls derived from the URDF collision meshes. Explicit limit-only results
say `joint_limits_only` and do not check collisions. Environment collisions,
dynamics, acceleration, calibration, tracking uncertainty and physical servo
interpolation are not modeled. See [SELF_COLLISION.md](SELF_COLLISION.md).

## Diagnostics

| Status | Meaning |
|---|---|
| `invalid_start` / `invalid_target` | Malformed input or violated input/configuration constraints. Target validation retains the existing one-meter input bound. |
| `ik_solution_not_found` | No sufficiently accurate, configuration-valid endpoint within the IK budget. |
| `roadmap_connection_not_found` | Valid goal exists, but start/goals cannot attach within the neighbor budget. |
| `path_not_found` | Valid endpoints exist, but graph components are disconnected or final trajectory validation failed. |
| `local_rollout_stalled` | Local NN stopped making enough progress. |
| `local_rollout_budget_exhausted` / `local_rollout_constraint_violation` | Local NN exceeded its budget or proposed an invalid transition. |
| `success` | Valid endpoint and validated joint plan, under the stated validation assumptions. |

With refinement/fallback enabled, unsuccessful local rollout reasons are retained
in `ik.attempts`; total endpoint failure is reported as IK failure. None of the
finite-budget failure outcomes proves physical unreachability.

Results include `waypoints_rad`, `waypoints_deg`, `fk_transforms_m`, `frames`,
`controller_waypoints_deg`, goal errors, path cost/length and IK/planning timings.
The command preview is not authorization or an executable hardware plan. Existing
hardware execution remains behind its own reviewed server-plan interface.

## Original limit-only results (before the self-collision upgrade)

Current self-collision results are saved separately in `benchmark_prm_self.json`.
The following measurements describe the original limit-only implementation.

`artifacts/benchmark_prm.json` records every case and fallback, model hash, runtime,
limits/environment and build configuration. It compares existing conditional NN
and seed+DLS rollouts, conditional/seed NN with refinement, numerical multiseed IK,
NN plus multiseed IK/PRM, and direct-only/PRM pairs with identical goal generation.
A numerical variant with eight random seeds matches the maximum 13 refinement
seeds available to the NN+fallback variant. Both stop after four accepted endpoints;
actual attempts and timings reveal differing work. Build time is separate; CPU
query timings exclude model loading and include preview trajectory generation.

The limit-only configuration box has no blocked joint paths. Therefore this real
case set measures **endpoint IK**, and cannot establish a routing improvement.
The saved synthetic routing experiment fixes the same FK-accurate endpoint for
both methods: a disk constraint blocks the straight path, PRM finds a validated
via path, and a complete separating wall remains disconnected. These establish
routing behavior, not real-arm collision performance. Reusing nodes as IK seeds
can improve convergence; this is separate from finding a path to a known goal.

Only 24 development cases and one training seed are used. Compare success,
endpoint errors, path travel and total timing before crediting the NN with value
over numerical IK. More seeds and a roadmap are extra computational work; neither
expands the robot's geometric workspace.

Recorded XYZ results (seed 9001, 24 cases, 1 mm; errors average all cases,
travel averages successful paths; timing varies by machine):

| Method | Success | Error mm | IK ms | Planning ms | Travel rad |
|---|---:|---:|---:|---:|---:|
| Existing conditional NN | 0/24 | 155.070 | 49.5 | 1.7 | — |
| Existing seed + DLS rollout | 24/24 | 0.916 | 9.7 | 0.3 | 1.808 |
| Conditional NN + refinement | 23/24 | 3.566 | 60.5 | 3.5 | 1.212 |
| Seed NN + refinement | 24/24 | 0.924 | 34.0 | 3.6 | 1.197 |
| Numerical multiseed + PRM | 24/24 | 0.923 | 30.6 | 3.7 | 1.222 |
| NN + PRM | 24/24 | 0.926 | 34.9 | 3.6 | 1.196 |
| Numerical matched seed budget + PRM | 24/24 | 0.923 | 30.7 | 3.7 | 1.222 |

NN+PRM has no demonstrated success-rate or speed advantage over numerical
multiseed IK here, although its selected endpoints have slightly shorter travel.
The existing single-seed DLS hybrid already solves these XYZ cases with less IK
work. `global-03` still fails with conditional-rollout refinement and no numerical
fallback. All successful real-case plans are direct connections. The 400-node
roadmap reuses 200 FK joint samples and has 3,030 edges; growth took about 55 ms
excluding construction/provenance loading. The synthetic PRM path has 4.923 rad
travel, while the same direct-only query cannot connect. The separating-wall query
reports `path_not_found`.

A separate pose5 comparison is saved in `artifacts/benchmark_prm_pose5.json`:
existing bounded DLS reaches 19/24, while both numerical multiseed direct-only and
PRM variants reach 24/24. They share identical goal generation and all paths are
direct, so this improvement is from additional IK seeds, not routing or a neural
model. Reproduce with:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.benchmark_prm \
  --task pose5 --collision limits --max-step-rad .08 --count 24 --samples 400 --seed 9001 \
  --output simulation/artifacts/benchmark_prm_pose5.json
```

Validation passed 47 simulation tests and 42 fake-hardware/API tests. The boundary
regression retains exact interpolated endpoints at operating limits. Build, grow
and query CLI commands were exercised. No physical movement was performed.
