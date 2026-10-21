# 5D Seed NN and joint-space PRM

The 5D task controls `joint6_flange` XYZ and its directed local Z axis in
`g_base`. Its accepted orientations are `R_target Rz(spin)` for arbitrary local
spin. The unit vector `u_target = R_target @ [0,0,1]` has three components and two
degrees of freedom; actual direction is `ARM.transform(q)[:3,2]`. Opposite axes
have 180° error. There is no full rotation-matrix or base-yaw acceptance test.
PRM constrains direction at goals only; intermediate joint motion may point
elsewhere. This upgrade performs no physical robot motion.

## Existing evidence and implementation

The inspected
`../output/pdf/mycobot_ik_prm_workflow_report.pdf` and
`../output/workflow/mycobot_xyz_ik_prm_corrected.png` describe the preserved XYZ
workflow. Actual interfaces are `planning.generate_goals/query/trajectory`,
`prm.Roadmap.plan`, `service.plan`, and `pose5.solve/task/axis_error`.
The existing five-row solver already implements directed tool-Z orientation,
antipodal handling, local-Z spin freedom and bounded DLS. It is reused.

FK is shared between NumPy serving, differentiable PyTorch training, data
labels and browser URDF rendering. Internal XYZ/joints use meters/radians;
UI/API use millimeters/degrees. The TCP is the flange with zero tool offset.
Target offsets retain the existing convention:
`resolved_XYZ = base_XYZ + R_start @ EEF_local_offset`, including starting wrist
spin. Direction is a separate base-frame vector and is not rotated by that
offset transform. Operating limits retain the recorded URDF/SDK/firmware
intersection and margin in `joint_limits.json`.

## Separate versioned checkpoint

`seed5.py` serves `artifacts/seed5_v1/models.json` through NumPy; PyTorch is needed
only for training. A checkpoint contains explicit schema/task/version, feature
names, architecture, normalization, bounded-output parameterization, URDF hash,
base/TCP/offset/units, joint limits and completed-training provenance.
Incompatible XYZ weights, changed FK/normalization and untrained weights are
rejected. Missing/incompatible weights return a visible diagnostic and no NN
proposals; independent numerical fallback remains available. Smoke-trained
weights are explicitly labeled and are stored below `seed5_v1/smoke/`, not at
the production checkpoint path.

The twelve inputs are `[XYZ / 0.3 m, unit tool-Z in base, q_normalized]`, where
`q_normalized = (q_current - joint_center) / joint_radius`. Two tanh hidden
layers feed 24 scalars, producing four six-joint proposals:
`clip(q_current + joint_radius * tanh(raw), lower, upper)`.
The existing nine-input XYZ Seed NN and all XYZ checkpoints are preserved.

## Data and objective

`seed5_data.py` samples the existing 75% nearby perturbation / 25% independent
uniform strategy, rejecting pairs outside operating limits, configured
self-collision validity, finite checks or representable proposal deltas.
Rejection changes the accepted mixture; metadata records accepted nearby
counts. Goal labels are full shared FK position and tool-Z column from valid
`q_goal`. Current and goal configurations must both pass conservative URDF
self-collision checking at 1 mm clearance. NPZ labels use meters/radians and
unit vectors; adjacent JSON records normalization, robot/frame hashes, validity,
sampling seeds and bounds. Loading rechecks labels against FK.

Each proposal receives a dimensionless task loss:
`position_weight * ||(FK_XYZ-target)/0.3 m||² + direction_weight * (1-dot(u_actual,u_target))`.
The direction minimum is uniquely aligned, with opposite axes costing twice the
direction weight. A configurable `travel_weight * ||q_proposal-q_current||²`
uses radian-squared joint displacement. The minimum across four proposals
trains at least one solving branch instead of averaging incompatible joint
labels. An optional separate best-of-four normalized joint-distance loss to
the FK witness (`witness_weight`) was tested as auxiliary supervision; it does
not add a spin/yaw constraint at inference or acceptance.

### Collision-aware training

The dataset already contains collision-free current and goal configurations.
The trainer now also audits predicted proposals with the exact URDF/FCL checker
and its recorded clearance. Every invalid head receives a normalized joint
distance penalty toward that sample's known safe FK witness. Safe proposals get
zero collision-repair loss, so this does not pull every alternative IK posture
toward one joint label. The best-of-four pose objective remains separate.

FCL supplies a detached validity mask; the repair loss supplies gradients toward
the safe witness. This is supervised collision repair, not a differentiable
signed-distance or shortest-escape calculation. It biases endpoint proposals;
it does not certify a connecting path or handle unmodeled external obstacles.
Runtime endpoint checks, local-path checks and PRM fallback remain required.

Default options are `--collision-weight 0.1 --collision-every 10
--collision-batch-size 64`. Audits cover all four heads for a subset of the main
random training batch every ten optimizer steps. The weight applies on audited
steps; the repair term averages over invalid heads only, so safe predictions do
not dilute rare unsafe examples. Validation checkpoint selection always applies
the same repair objective to a larger fixed audit (`--collision-validation-batch-size
512`); final raw validity metrics cover the entire validation split.
Metadata records checker/geometry/clearance provenance, audit counts, the loss
definition, raw invalid-proposal fraction, tasks with a valid proposal, and
invalid pose-best proposals. A positive collision weight requires a
self-collision-validated dataset. `--collision-weight 0` reproduces the legacy
objective for a matched control.

`--init-checkpoint` can fine-tune a compatible NumPy checkpoint and records its
hash and completed training steps. `--learning-rate` controls the optimizer.
With collision training enabled and no explicit output, the CLI writes a
separate `candidates/collision-aware` directory (or `collision_aware_smoke`),
leaving the production checkpoint in place. Existing architecture, input
features, inference schema, XYZ weights and dataset files remain compatible.

Reporting and acceptance use the robust angular error
`atan2(norm(cross(u_actual,u_target)), dot(u_actual,u_target))`, in degrees.
Position and direction tolerances default to 1 mm and 1°. Cross magnitude alone
is never used as an acceptance error.

## Reproduce training

Run from the project root. The GPU run on this PC used the RTX 4070 Laptop and
PyTorch 2.14.1+cu130; training is deterministic with recorded seeds and a separate
validation split. Serving does not require CUDA. CPU installation:

```bash
.venv/bin/python -m pip install -r simulation/requirements-training.txt
.venv/bin/python -m pip install -r simulation/requirements-collision.txt
```

Reproduce the collision-repair fine-tune from the saved pre-activation checkpoint,
then compare exact proposal validity and refined direct paths on fresh FK tasks:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.train_seed5 \
  --dataset simulation/artifacts/seed5_v1/dataset_large.npz \
  --init-checkpoint simulation/artifacts/seed5_v1/backups/pre-collision-webui-20261004-574a9107421b/models.json \
  --output simulation/artifacts/seed5_v1/candidates/collision-aware-s102-balanced \
  --device cuda --hidden 384 --steps 4000 --seed 102 --batch-size 512 \
  --learning-rate 0.0001 --travel-weight 0.00001 --witness-weight 0.03 \
  --collision-weight 0.1 --collision-every 10 --collision-batch-size 64
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.benchmark_collision_training \
  simulation/artifacts/seed5_v1/backups/pre-collision-webui-20261004-574a9107421b/models.json \
  simulation/artifacts/seed5_v1/candidates/collision-control-s102/models.json \
  simulation/artifacts/seed5_v1/candidates/collision-aware-s102-balanced/models.json \
  --count 128 --seed 20261004 --output output/collision_training/comparison.json
```

Use identical initialization, seed, steps and optimizer settings with
`--collision-weight 0` in another output directory for a matched control.
These commands create candidates; production activation is a separate step.
The older reproduction commands below explicitly disable the collision term
to reproduce their historical loss.

GPU installation matching the installed driver on this PC:

```bash
.venv/bin/python -m pip install --upgrade torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
```

Original-size data and three candidates:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.seed5_data \
  --output simulation/artifacts/seed5_v1/dataset.npz --count 24000 \
  --validation-count 2048 --seed 42 --validation-seed 43 --collision self
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.train_seed5 \
  --device cuda --hidden 256 --steps 8000 --seed 42 --travel-weight 0.00001 --collision-weight 0 \
  --output simulation/artifacts/seed5_v1/candidates/w256-s42
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.train_seed5 \
  --device cuda --hidden 384 --steps 16000 --seed 52 --travel-weight 0.00001 --collision-weight 0 \
  --output simulation/artifacts/seed5_v1/candidates/w384-s52
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.train_seed5 \
  --device cuda --hidden 384 --steps 24000 --seed 62 --travel-weight 0.00001 --collision-weight 0 \
  --witness-weight 0.1 --output simulation/artifacts/seed5_v1/candidates/w384-s62-witness
```

Larger-data candidate and production selection:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.seed5_data \
  --output simulation/artifacts/seed5_v1/dataset_large.npz --count 96000 \
  --validation-count 2048 --seed 142 --validation-seed 43 --collision self
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.train_seed5 \
  --dataset simulation/artifacts/seed5_v1/dataset_large.npz --device cuda \
  --hidden 384 --steps 32000 --seed 72 --travel-weight 0.00001 --witness-weight 0.03 --collision-weight 0 \
  --output simulation/artifacts/seed5_v1/candidates/w384-s72-large
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.select_seed5 \
  simulation/artifacts/seed5_v1/candidates/w256-s42 \
  simulation/artifacts/seed5_v1/candidates/w384-s52 \
  simulation/artifacts/seed5_v1/candidates/w384-s62-witness \
  simulation/artifacts/seed5_v1/candidates/w384-s72-large --count 512
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.benchmark_seed5 \
  --count 24 --seed 20261002 --output simulation/artifacts/seed5_v1/benchmark.json
```

Selection uses a fixed 512-task validation subset with numerical fallback
**disabled**, maximizing collision-valid success within both tolerances and
breaking ties by total refinement iterations. It publishes only the selected
full-trained candidate to the separate production path. `selection.json`
records all candidate results; no final held-out task influences selection.
`benchmark.json` records numerical-current, numerical-multiseed, raw NN and
NN-refined planning, with individual tasks/errors/timings/outcomes and two
same-XYZ/different-direction families tested from different starts, plus eight
zero/rest-start tasks. Same-XYZ family witnesses are generated using the existing
5D solver at 0.1 mm / 0.1°; labels retain identical XYZ and actual FK directions.

A small CPU smoke check (not full training):

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.seed5_data \
  --output simulation/artifacts/seed5_v1/smoke/dataset.npz \
  --count 128 --validation-count 32 --seed 42 --validation-seed 43
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.train_seed5 \
  --dataset simulation/artifacts/seed5_v1/smoke/dataset.npz \
  --smoke --hidden 128 --device cpu --steps 8 --batch-size 32
```

## Endpoint queue and planning

1. Validate current joints and target; normalize direction. Already-at-target
   requires finite/limit/collision validity and **both** tolerances.
2. Check all four raw NN guesses. Collect unique accepted configurations.
   Independently queue every finite in-limit guess when refinement is enabled,
   including accepted guesses and initially self-colliding guesses.
3. Before refinement, if no raw goal passed and numerical fallback is enabled,
   append current, up to four pose-near roadmap configurations and four
   deterministic random configurations. Fallback is independent of NN refinement.
   Rank roadmap seeds by `XYZ_distance/0.3 m + 0.6*axis_angle_radians`.
4. Preserve NN → current → roadmap → random order; skip duplicate seeds.
   Refine until queue exhaustion or four unique accepted goals by default.
   Validate every endpoint with both tolerances and existing validity checks.
   Iterations are computations, not executed robot motion.
5. Try valid direct joint connections first; otherwise attach start/goals to
   the reusable PRM and search. Preserve edges, optional shortcuts, final path
   checking and speed-limited validated interpolation. Known joint waypoints
   do not invoke IK again. IK success does not guarantee route feasibility.

The WebUI's 5D PRM button and mouse aim additionally enable the bounded
[seed/solver recovery stage](IK_RECOVERY.md): extra coverage only after IK
failure, or one expansion to eight distinct goal candidates after routing
failure. The normal queue above, XYZ mode and trained checkpoints are preserved.
Python/API callers opt in with `pose5_global_recovery=True`.

Result fields distinguish `ik_outcome` and `planning_outcome`, with errors,
`target_axis`, `achieved_axis`, raw accepted count, checkpoint diagnostic,
accepted-goal sources and inference/refinement timings. The WebUI provides
base-vector entry, useful axis/current-tool presets, legacy reference tilts,
visible mm/degree tolerances, separate 5D Seed NN / numerical-only proposals,
and achieved direction. Existing four numerical controller comparisons and
XYZ mode remain available.

Simulation mouse aim first attempts a complete, collision-checked local IK path
from current joints, then uses this trained 5D pipeline as fallback through
`/api/sim/mouse-aim`, with a camera-facing cursor plane, 50 mm standoff,
self-collision-checked direct/PRM paths and validated joint playback. See
[MOUSE_AIM.md](MOUSE_AIM.md) for geometry, coalescing/cancellation, endpoint
acceptance and simulation-only gating.

The existing `artifacts/roadmap.json` is reused after robot/environment/geometry
metadata validation. Changing only goal orientation does not rebuild joint-space
connectivity. New workflow sources:
`../output/workflow/mycobot_seed5_ik_prm.{mmd,dot,svg,png}`. Render the graph with
`dot -Tsvg ...dot -o ...svg` or `dot -Tpng -Gdpi=115 ...dot -o ...png`.

## Validation and measured results

The collision-repair experiment fine-tuned the production checkpoint for 4,000
steps on the existing 96,000-pair dataset. A matched control used the same
initial weights, seed 102, batch size, learning rate and optimizer schedule,
with collision weight zero. The candidate used the defaults above and was
selected at step 3,601 by the audited validation objective. All four heads were
audited on 102,400 training predictions, including 1,257 unsafe predictions.

| Measurement | Matched control | Collision-repair candidate |
|---|---:|---:|
| Unsafe raw proposals, 2,048 validation tasks | 138/8,192 (1.68%) | 59/8,192 (0.72%) |
| Mean pose-best validation position / direction error | 17.40 mm / 5.60° | 17.93 mm / 5.72° |
| Unsafe raw proposals, 128 fresh FK tasks | 9/512 (1.76%) | 5/512 (0.98%) |
| Refined valid endpoints / collision-checked direct paths | 128/128 / 128/128 | 128/128 / 128/128 |
| Mean refinement iterations on fresh tasks | 118.58 | 126.77 |

Fresh tasks used seed 20261004 and numerical fallback was disabled. No raw
proposal solved those tasks within 1 mm and 1°; refinement remained necessary.
The lower unsafe-proposal fraction came with a small raw pose accuracy tradeoff
and more refinement work. This is one training seed and a finite FK sample,
not a collision-free guarantee. The candidate is retained in
`artifacts/seed5_v1/candidates/collision-aware-s102-balanced/`. The experiment
initially preserved the production checkpoint, original datasets and roadmap.
Full
measurements are in
[the comparison report](../output/collision_training/comparison.json) and
[validation provenance](../output/collision_training/validation.json).

On 2026-10-04, the user's subsequent WebUI update request activated this candidate
at `artifacts/seed5_v1/models.json` with matching training metadata and PyTorch
weights. Both PRM goal proposals and the mouse-aim global fallback now use it;
mouse aim still tries local IK first. Status fields include the loaded checkpoint
hash, fine-tuning flag and self-collision training metadata. WebUI labels identify
the active model without implying that training certifies motion safety.
`artifacts/seed5_v1/activation.json` records this explicit activation and the
previous-model backup at
`artifacts/seed5_v1/backups/pre-collision-webui-20261004-574a9107421b/`.
The earlier `selection.json` and benchmark remain historical measurements of
the previous checkpoint. XYZ weights, datasets and roadmap remain unchanged.

Final checkpoint selection and held-out measurements are recorded in the
adjacent training/selection/benchmark JSON files and [SEED5_RESULTS.md](SEED5_RESULTS.md).
The 8-step smoke run solved **0/32** raw tasks within 1 mm and 1°; it verified
training/export/serving only and is not used as a production model.

Checks cover direction validation, local-Z spin invariance, opposite rejection,
position-correct/direction-wrong targets, the dual-tolerance shortcut, all-four
raw acceptance, independent refinement eligibility including colliding seeds,
fallback condition/order/uniqueness/budget, target-offset frames and XYZ/checkpoint
compatibility. Browser checks cover vector normalization, invalid-vector request
blocking, source/outcome displays, preview, XYZ preservation and mouse-aim mode.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s simulation/tests -v
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s webui/tests_python -v
cd webui
npx playwright test tests/seed5.spec.js tests/pose5.spec.js tests/prm.spec.js tests/mouse-aim.spec.js
npm run build
```

## Limits

A seed NN approximates IK; tight raw acceptance can remain rare after full
training, so refinement is part of its intended workflow. This is the best of
the evaluated candidates on a finite validation sample, not a global optimum
or guarantee for all reachable tasks. FK-generated examples are biased by the
sampling/proposal bounds; out-of-distribution tasks and unreachable pointing
combinations can fail. Goals and trajectories check conservative URDF convex
self geometry; external obstacles, cables, attached tools, calibration and
physical dynamics are not modeled. Intermediate orientation is unconstrained.
No physical execution or hardware performance test was performed.

## Source locations for the inspected workflow

These are actual locations in the implementation after the upgrade:

| Responsibility | Source |
|---|---|
| Shared flange transform / differentiable FK | `kinematics.py:70`, `kinematics.py:100`, `kinematics.py:120` |
| Unit-axis validation / robust directed angle | `pose5.py:64`, `pose5.py:90` |
| Twelve-feature schema / bounded NumPy proposals / checkpoint guard | `seed5.py:13`, `seed5.py:34`, `seed5.py:43` |
| FK-labeled valid representable sampling | `seed5_data.py:12`, `seed5_data.py:31` |
| Differentiable unique-direction task loss / best-of-four training | `train_seed5.py:19`, `train_seed5.py:60` |
| Exact collision mask / differentiable witness repair | `train_seed5.py:27`, `train_seed5.py:54` |
| Fresh FK collision/refinement comparison | `benchmark_collision_training.py:15` |
| Both-tolerance goal acceptance / all raw checks / independent queues | `planning.py:62`, `planning.py:100` |
| Direct/PRM planning and separate failure outcomes | `planning.py:233`, `prm.py:239` |
| Speed-limited interpolation, no waypoint IK | `planning.py:199` |
| Full starting-EEF rotation for local target offset | `service.py:32` |
| Vector normalization, presets and UI entry | `../webui/src/tool-direction.js:4`, `../webui/src/tool-direction.js:34` |

The original report and corrected XYZ diagram remain unchanged. Their accurate
queue/acceptance/planning sequence is retained; the 5D diagram adds the separate
checkpoint, directed-axis inputs and dual-tolerance endpoints. The report's
XYZ-only NN cannot be reinterpreted as orientation-aware without this new data,
training and checkpoint schema.
