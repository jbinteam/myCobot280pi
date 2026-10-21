# myCobot diffusion motion-planning experiment

This implements the five phases in the project's `research.md` using the actual
myCobot 280 Pi URDF, six installed joints and recorded operating limits.
The experiment runs in simulation and learns joint trajectories from expert
RRT* demonstrations. Its trained policy is available alongside the WebUI's
active Seed5/PRM and numerical controllers.

For a fresh build or another robot arm, follow the
[diffusion build and porting procedure](docs/BUILD_AND_PORT_DIFFUSION.md).
It covers robot/schema adaptation, checked demonstrations, GPU training,
evaluation and robot-specific deployment.

## WebUI preview

Start `./start_webui.sh` and open http://127.0.0.1:8000. Under **Diffusion
trajectory planner**, select **Block detour**, **Nearby goal**, or **Global
goal**, then press **Plan checked diffusion**. These are held-out tasks;
each request generates a fresh policy rollout rather than replaying stored
evaluation frames. **Play** and the time slider inspect the validated result.
Block position and dimensions, sampling seed, candidate count, denoising
steps and decision budget are editable. For custom Cartesian targets, select
the 5D task; endpoint IK tries current joints first and checks the same scene.

**Obstacle → No block** removes only the optional box, retaining the table,
self-collision and joint-limit checks. It applies to full diffusion previews and
local-first diffusion mouse tracking. `obstacle_mm` must be explicitly supplied:
six block geometry numbers or JSON `null`. Missing geometry is rejected rather
than interpreted as permission to remove an obstacle. Responses report
`scene_mode` (`block` or `no_block`), the active `collision_checks`, and optional
`conditioning_note`. The existing `joint_limits_self_table_block` validation
profile denotes the scene checker with an optional block. Browser tracking and
full previews match the returned scene to the requested one before playback.

The original observation schema supports no block by zeroing the six geometry features
and the presence mask. The original v1 checkpoint was trained with blocks, so this
condition was experimental before v2 training. Three fresh v1 no-block rollouts using the held-out
presets all failed safely (two `no_safe_chunk`, one `decision_limit`), returning
no playable frames. These probes establish a current limitation, not a general
success-rate estimate. Local IK remains usable for checked nearby mouse chunks;
reliable diffusion planning needs absent-block demonstrations and retraining.
The original v1 dataset/checkpoint remain preserved; the active v2 model and
training workflow are documented below.
See the [live no-block API, continuous collision replay and artifact audit](../../output/diffusion/no_block_deployment_validation.json).

`service.py` exposes checkpoint/scene metadata through `/api/sim/info` without
loading Torch or Bullet. `POST /api/sim/diffusion` lazily loads the pinned full
checkpoint, serializes its mutable scheduler/generator, seeds each request,
and creates a separate Bullet scene. `artifacts/diffusion_active.json` selects
the pinned manifest (v2 currently), while `artifacts/diffusion_v1/webui.json`
preserves the original deployment. Each manifest pins the model hash, training
scene and three task presets. Replacing weights
requires an explicitly updated deployment manifest. A missing or incompatible
diffusion installation leaves the original controllers available.

The adapter rejects unsafe starts/goals and every unsafe full 16-step chunk.
V1 advances two speed-projected references per decision. V2 fits a checked
full-horizon C2 curve and advances up to 0.4 seconds with matching boundary
velocity/acceleration, retaining a previous checked braking tail when a new
proposal is rejected. It then checks the whole
completed path before smoothing and returning frames. Failed or exhausted
rollouts return no playable partial path. Budgets are at most 200 decisions,
with a 90-second deadline checked between decisions. The final exact goal
capture is reported separately. Collision-checked shortcuts and monotone quintic
timing produce final 0.02-second references, with zero velocity and acceleration
at the start, endpoint and retained corners. Every timed state and connecting
edge is checked again. Input edits discard pending results without moving the viewer.

The adapter also exports checked segment endpoints and their exact rounded
duration. Browser playback verifies they reproduce every returned reference,
checks analytic speed/acceleration bounds, and evaluates their quintic at each
render time. It preserves corners, start/end rest and pause/resume timing. Blue
TCP output dots, a cyan current-reference marker and a joint velocity plot expose
the trajectory being rendered. The 100,000-episode dataset is unchanged by this
rendering update.

The scene shows the same 0.9 m table (top −35 mm) and axis-aligned block used
by collision checking. Other planner buttons hide this scene. This route is
available only when hardware is disabled and disconnected, with no active
robot motion. It never creates a hardware execution plan. Speed is bounded
to 1 rad/s and acceleration to 3 rad/s² on the planned references.

The optional WebUI mouse engine **Local IK → diffusion · self / table / block**
uses `tracking.py` through the guarded `/api/sim/mouse-aim` route. It reuses one
isolated Bullet scene under the shared inference lock, validates local IK
against that scene first. With Continuous motion off, it selects the next two diffusion references or up to
0.2 seconds of local IK, then smooths and retimes that prefix to at most 1.3 seconds. A
playable prefix does not imply convergence; the browser replans stationary and
moving cursors, preserves each cursor's target axis and discards stale results.
Normal cursor changes wait for the current chunk to finish at rest; Stop,
Escape and leaving the scene can interrupt immediately. Previously selected
IK goals are revalidated hints, never raw motion.

The default **Continuous motion** simulation mode uses a complete local path or
the full selected checked diffusion horizon instead of the short resting prefix
described above. Requests add `continuous_motion: true`,
`start_velocity_rad_s` and `start_acceleration_rad_s2` for a predicted future
reference. `simulation/continuous.py` fits C2 Bezier segments with those starting
derivatives and a resting endpoint. Convex control hulls enforce joint limits;
subdivided derivative hulls bound speed and acceleration over the entire curve.
Each new curve and connecting sample edge is collision checked at 100 Hz, with
no unchecked blending fallback. References use 0.02-second timing, a 12-second
route budget, speed ≤1 rad/s and acceleration ≤3 rad/s². Retained corners stop.

The browser keeps rendering the old checked curve while computing one replacement
ahead of playback. It validates the returned curve, scene and boundary motion,
queues the replacement for its exact future reference, and preserves position,
velocity and acceleration at the splice. Moving cursor snapshots may be followed
briefly while the latest cursor is coalesced for the next query. An expired
interior splice is discarded; the old curve remains available to brake to rest.
Only a matching resting endpoint may accept a late resting-start plan. Stop and
scene changes invalidate both the active curve and its queued replacement.
This mode also works with the classical self-checked simulation mouse planner;
hardware mouse tracking and full diffusion preview behavior are unchanged.
See the [live continuity, collision replay and artifact audit](../../output/diffusion/continuous_mouse_deployment_validation.json).

The original v1 checkpoint has no target velocity/history conditioning or moving-target
demonstrations. V2 adds current joint velocity/acceleration, target velocity
and moving-goal demonstrations. The six legacy cursor replays in `benchmark_tracking.py` are an
experimental integration check, not a tracking reliability estimate. Some
targets fail or stall. Full-path previews
retain their original all-or-nothing contract. See [mouse aim](../MOUSE_AIM.md).

Integration tests:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest simulation.diffusion.tests.test_service simulation.diffusion.tests.test_tracking
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.diffusion.benchmark_tracking
cd webui
npx playwright test tests/diffusion.spec.js tests/diffusion-tracking.spec.js
```

## Motion-conditioned v2 deployment

The active checkpoint is `artifacts/diffusion_motion_v2/model/diffusion_policy_6dof.pth`.
It starts from the preserved v1 EMA weights, expands the observation from 19 to
34 features, and fine-tunes for 20,000 AdamW steps on the 100k expert dataset
plus 20k augmentations. Extra features are six joint velocities (scale 1 rad/s),
six joint accelerations (scale 3 rad/s²), and three Cartesian target velocities
(scale 0.5 m/s). The browser estimates target velocity from successive target
snapshots; it passes the analytical future-reference joint motion at a join.
The robot/geometry identity remains the original schema; a separate versioned
`conditioning` object pins the new network input layout.

The augmentation contains 7,274 static no-block routes, 5,454 moving no-block
routes, 3,636 moving block routes and 3,636 nonzero-motion retargets (half without
a block). Moving targets look ahead along checked expert routes; retargets join
a new valid goal from incoming motion using collision-checked C2 curves.
Removing a block permits checked shortcuts; one retimed shortcut failed its
collision check and retained the original route. All parent-child pairs inherit
the same split. Combined train/validation/test counts are 95,916 / 11,992 / 12,092.
Training samples original/augmented episodes equally, then samples a frame
uniformly within an episode. Training and EMA selection load train/validation
arrays; test arrays are excluded. The selected EMA step is 20,000, validation
noise MSE is 0.001521, and training took 207 seconds after cache preparation.
The original 1k/100k datasets, v1 checkpoint, URDF and classical artifacts remain
unchanged. All 20k augmented episodes passed split/shape/motion audits, with an
independent collision replay of 1,000 episodes and 46,898 states.

Physically small first predictions from rest do not work with the former
two-reference rollout: that loop stalled on this model. V2 full previews use
`continuous_rollout.py`: check the full proposed horizon, fit the same C2 curve
used by continuous mouse tracking, advance at most 0.4 seconds at 0.02-second
references, and condition the next prediction on the resulting motion. The
existing checked braking tail remains available after rejected replacements.
Final previews still receive whole-path collision checks and checked quintic
playback. With Continuous motion off, hybrid tracking extracts a bounded
geometric prefix from the checked horizon before retiming it into a resting chunk.

The paired comparison uses the **same full-horizon C2 controller for both models**,
four candidates and 32 denoising steps. Neural suites bypass local IK so the
results measure the diffusion component rather than local solver success.

| Held-out simulation measurement | Original v1 | Motion v2 |
|---|---:|---:|
| Fixed goals, 64 tasks | 29/64 | 63/64 |
| No-block fixed goals | 0/32 | 32/32 |
| Direct-blocked fixed goals | 13/16 | 15/16 |
| Moving/retarget goals, 24 tasks | 11/24 | 23/24 |
| Moving-target TCP RMSE over 48 s | 42.0 mm | 31.7 mm |
| Moving-suite stalled/rejected/no-progress decisions | 164 | 134 |
| Moving-suite maximum sampled jerk | 114.3 rad/s³ | 88.6 rad/s³ |
| CPU denoising median / p95 | 63.2 / 63.5 ms | 64.0 / 64.3 ms |
| Executed collisions | 0 | 0 |

RMSE uses equal durations, including the low-error hold after early success;
averaging only until each model finishes would penalize faster completion.
Reference speed/acceleration remained within 1 rad/s and 3 rad/s², and analytical
velocity/acceleration matched at all joins. These are 88 scenarios and one
training seed, with moving goals restricted to expert routes. Tracking metrics
exclude planner compute delay, and CPU timings measure inference only. They
are not browser-lag or physical-arm reliability estimates. Raw unsafe proposals
remain possible and are counted separately from executed collisions. All three
WebUI presets reached their goals both with and without the block.

Evidence: [dataset audit](../../output/diffusion/motion_v2_dataset_validation.json),
[training provenance](../artifacts/diffusion_motion_v2/model/training.json),
[paired comparison](../../output/diffusion/motion_v2_continuous_comparison.json),
[six full preview checks](../../output/diffusion/motion_v2_full_preview_validation.json),
[live C2/FK/collision replay and artifact audit](../../output/diffusion/motion_v2_deployment_validation.json).

The completed workflow used these commands (choose fresh output/cache paths to
generate or train again; existing outputs are protected):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m simulation.diffusion.generate_motion
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m simulation.diffusion.audit_motion
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m simulation.diffusion.train_motion --steps 20000
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m simulation.diffusion.compare_continuous --only baseline
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m simulation.diffusion.compare_continuous --only candidate
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m simulation.diffusion.compare_continuous
.venv/bin/python -m simulation.diffusion.deploy_motion
```

Deployment requires matching model/data/evaluation hashes and every paired
comparison check. `artifacts/diffusion_active.json` atomically selects the v2
manifest. Removing that pointer and restarting the WebUI restores the original
v1 manifest/checkpoint; no original file needs replacement.

## Robot and environment

`env.py` loads the shared URDF in a separate PyBullet client. COLLADA meshes are
converted to the same conservative convex hulls used by the project's native
self-collision checker. Joint/mesh origin transforms and meter scaling are
preserved. Converted OBJ/URDF files have a hash manifest and a process lock;
changed cached geometry is rejected. The original URDF is never edited.

The scene has a fixed table and one variable axis-aligned block on the table.
Self-collision uses the existing FCL profile; robot/table and robot/block
clearance use Bullet's convex geometry queries. Checks require 1 mm clearance
and sample connecting joint paths at at most 0.04 rad per joint. This is finite
sampling, not a continuous-collision certificate.

The flange TCP is `joint6_flange` in `g_base`. `forward_kinematics` uses the
shared transform. `inverse_kinematics` finds a valid endpoint for XYZ and a
directed tool-Z vector, preserving free local-Z spin. The planner and policy
then condition on the selected goal joints. Dataset Cartesian labels retain
the same five-dimensional convention.

Execution resets checked joint references in PyBullet. The URDF has no
calibrated dynamics; loader inertias are explicit placeholders, gravity is
disabled, and no robot transport is imported. This does not validate real
motors, forces, tracking delay or contact dynamics.

## Expert demonstrations

`planner.py` implements seeded joint-space RRT* parent selection and rewiring,
including descendant cost propagation and cycle prevention. Its Euclidean
metric scales by operating joint range. Bounded joints never wrap at ±π.
A checked direct connection attains the straight-line lower bound; blocked
connections use the finite-budget tree search. Neighbor limits and iteration
budgets make this a practical finite variant, not an optimality claim.

Smoothing uses newly collision-checked shortcuts. Monotone quintic timing
follows those segments, stops at corners and bounds vector speed/acceleration
to 1 rad/s and 3 rad/s². It avoids unchecked splines cutting through obstacles.
Every actual timed frame and connecting edge is checked again. The initial
full run exposed a grazing collision missed by a different sampling grid;
the generator now rejects such trajectories and preserves a regression fixture.

`generate_dataset.py` records HDF5 episodes with joint trajectories, timestamps,
goal joints, goal XYZ/tool-Z, obstacle center/half-extents and planner diagnostics.
Splits are stratified by nearby/global and direct/blocked task type at episode
level. Windows from one trajectory cannot leak into another split.

### New 100,000-episode dataset

`generate_large.py` generates independent, resumable shards and merges them into
one HDF5 dataset. The new run uses seed 20261005, 12 workers, 100 shards of 1,000
episodes and a 20% blocked-route quota. Seeds, source hashes and robot/scene
identity are recorded before generation. Existing datasets and weights are retained.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m simulation.diffusion.generate_large \
  --output simulation/artifacts/diffusion_100k_v1/demonstrations.h5 \
  --episodes 100000 --seed 20261005 --workers 12 --shard-size 1000 --audit-episodes 1000
```

To continue an interrupted run, use the same command with `--resume`. Completed
shards are verified before reuse. `demonstrations_shards/progress.json` reports
durably completed episodes; the final HDF5 appears only after merge and audit.
Generation checks every timed path for collision. Merge checks all episode
shapes, joint limits, endpoints, FK labels, timing, speed and acceleration,
creates global stratified train/validation/test splits, and independently
replays collisions on 1,000 selected episodes. The adjacent `demonstrations.json`
records the final checksum, counts, validation and provenance. The WebUI
checkpoint originally remained trained on the 1,000 episodes; the v2 run below
now uses this large dataset with motion/absence augmentation.

The completed run contains **100,000 episodes**, 5,328,757 timed reference states,
20,012 blocked-route demonstrations and 58,810 nearby-goal tasks. Global episode
splits contain 79,998 train, 9,998 validation and 10,004 test episodes. Generation,
merge and collision replay took 2,788 seconds; 793 planner-budget failures and
291 trajectories failing their final collision recheck were excluded.
The final HDF5 is 890,713,465 bytes, with SHA-256
`d0818df88bee27b9a55c2168340eba5f6ac1f5f4c69181e2439a007861a47ab3`.

- [100,000-episode dataset](../artifacts/diffusion_100k_v1/demonstrations.h5)
- [Generation provenance and collision audit](../artifacts/diffusion_100k_v1/demonstrations.json)
- [Reopened file and unique-source validation](../../output/diffusion/dataset_100k_validation.json)
- [Deployed smoothing verification](../../output/diffusion/smoothed_deployment_validation.json)

## Diffusion policy

`dataset.py` constructs future-position windows `[6,16]`. Action zero is the
next state, not the current state. Short tails repeat the terminal pose.
Normalization uses fixed operating joint bounds, independent of held-out data.

Observations `[19]` contain six current normalized joints, six selected goal
normalized joints, and seven block features: scaled XYZ, scaled half-extents,
and presence. The constant table is part of the environment identity.

`model.py` provides a custom temporal Conv1d U-Net with channels 32/64/128,
skip connections, sinusoidal diffusion time and observation-conditioned FiLM
residual blocks. It maps noisy actions `[batch,6,16]` to predicted noise of the
same shape. This is a small project-specific architecture, not a reproduction
of the published paper's benchmark configuration.

`train.py` uses the Diffusers DDPM scheduler, 100 cosine noise steps, Gaussian
noise MSE, AdamW, cosine learning-rate decay, EMA and early stopping. Fixed-seed
validation noise selects the EMA checkpoint. Test episodes are never loaded by
training. Weights load with `weights_only=True`; schema, geometry, joint limits,
training status and finite weights are checked before inference.

## Original v1 receding-horizon evaluation

`policy.py` denoises Gaussian chunks and proposes several alternative paths.
`eval.py` batches denoising across isolated scenes. It rejects unsafe raw paths,
projects accepted paths to the speed limit, then checks the complete projected
chunk and connecting edges. At most the first two references are executed
before observing the current joints and replanning.

If the exact goal is within one speed-bounded step, a checked final goal
connection completes the rollout. This conventional endpoint capture is
recorded separately. RRT* is only an expert baseline; it never repairs a policy
rollout. Failed safe-chunk searches hold the pose and are counted as failures.
Execution collision rate and rejected raw-proposal rate are separate metrics.
This offline evaluator measures acceleration without bounding it. Its historical
results below precede the separate final smoothing now used by the WebUI;
the WebUI retimes accepted paths with the 3 rad/s² reference limit before playback.

## Original v1 measured run

| Measurement | Result |
|---|---:|
| Accepted expert episodes | 1,000 |
| Demonstrations requiring detours | 200 |
| Stored reference states | 53,543 |
| Train / validation / test episodes | 799 / 99 / 102 |
| Training windows | 42,638 |
| U-Net parameters | 526,054 |
| Training epochs / selected EMA epoch | 53 / 41 |
| Best fixed validation noise MSE | 0.008115 |
| Held-out RHC successes, 1 mm and 1° | 44/50 (88%) |
| Blocked-route successes | 6/11 |
| Expert baseline successes | 50/50 |
| Unsafe raw chunks rejected | 838/5,456 (15.36%) |
| Episodes with unsafe raw proposals | 13/50 |
| Executed collisions | 0/50 |
| Successful rollouts using final endpoint capture | 44/44 |
| Maximum reference speed / measured acceleration | 1.0 rad/s / 17.26 rad/s² |

Generation used seed 42 and took 134 s; eight tree-budget failures and three
timed-path rejections were excluded. Training used seed 42 and the RTX 4070
Laptop GPU, took 52 s, and early-stopped after 4,452 optimizer steps. Evaluation
used test-selection/noise seed 20261004, 32 denoising steps, four sampled chunks,
and at most 200 decisions. It took 54 s. Five blocked tasks failed to find a
safe chunk; one other task exhausted the decision budget. No hyperparameters
were changed using these held-out results.

Zero executed collisions includes the runtime safety gate's effect and does
not mean the learned policy is collision-free. Blocked-route coverage remains
weaker than the expert planner. The measured acceleration exceeds the expert
training bound, so these are kinematic reference results, not hardware-ready
trajectories. Results cover one seed and the recorded table/block family.

Evidence:

- [Phase 1 FK/environment verification](../../output/diffusion/phase1.json)
- [Dataset provenance](../artifacts/diffusion_v1/demonstrations.json)
- [Training provenance](../artifacts/diffusion_v1/model/training.json)
- [Per-episode evaluation and executed joint traces](../../output/diffusion/evaluation.json)
- [Validation record](../../output/diffusion/validation.json)
- [Checked rollout preview](../../output/diffusion/preview/preview.json)

## Run it

Use the project's existing Python 3.12 environment. The dependency file includes
the native collision packages. Keep the existing CUDA-enabled Torch build;
PyBullet may build its wheel locally. Exact installed versions are recorded in
`output/diffusion/dependencies.json` and training/environment metadata.

```bash
.venv/bin/python -m pip install -r simulation/diffusion/requirements.txt
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.diffusion.env
```

Choose an unused output directory; dataset/model commands refuse to overwrite
completed artifacts:

```bash
DIFFUSION_RUN=simulation/artifacts/diffusion_v1/reproduction
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.diffusion.generate_dataset \
  --output "$DIFFUSION_RUN/demonstrations.h5" --episodes 1000 --seed 42
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.diffusion.train \
  --dataset "$DIFFUSION_RUN/demonstrations.h5" --output "$DIFFUSION_RUN/model" \
  --device cuda --epochs 80 --seed 42
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.diffusion.eval \
  --dataset "$DIFFUSION_RUN/demonstrations.h5" \
  --checkpoint "$DIFFUSION_RUN/model/diffusion_policy_6dof.pth" \
  --output "$DIFFUSION_RUN/evaluation.json" --episodes 50 --device cuda
```

Evaluate the included trained checkpoint, or display one held-out rollout:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.diffusion.eval --device cuda
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.diffusion.eval \
  --gui --episodes 1 --device cuda --output output/diffusion/gui_episode.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.diffusion.preview
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s simulation/diffusion/tests -v
```

## Cartesian conditioning without runtime IK

The `diffusion_cartesian_v3` candidate replaces the six goal-joint input values
with target XYZ and a directed flange local-Z unit vector. It predicts joint
trajectories directly from the current arm state and target pose. The local-Z
spin remains free, matching the existing five-dimensional pose task.

| Feature positions | Inputs | Normalization |
|---|---|---|
| 0–5 | Current joint angles | Operating joint range |
| 6–8 | Target XYZ in `g_base` | Position / `ARM.scale` |
| 9–11 | Target directed flange-Z axis | Unit vector |
| 12–18 | Block center XYZ, half-extents XYZ, presence | Geometry / `ARM.scale`; all seven zero when absent |
| 19–24 | Current joint velocity | 1 rad/s vector bound |
| 25–30 | Current joint acceleration | 3 rad/s² vector bound |
| 31–33 | Target Cartesian velocity | 0.5 m/s vector bound |

The absolute Cartesian input above uses **schema 3**. The relative Cartesian
variant uses **schema 4**: positions 6–8 contain target XYZ minus current TCP
XYZ, and positions 9–11 contain the directed target axis minus the current
flange-Z axis. The encoder computes the current pose with FK. These explicit
errors make a reached target correspond to zero target features and remove the
need for the observation network to learn FK before interpreting the target.
Both variants retain 34 values and require no runtime IK or goal joints.

Input length alone cannot distinguish joint-goal, absolute Cartesian and
relative Cartesian conditioning. Model
loading and runtime dispatch check the complete versioned conditioning record.
Absolute joint-action chunks retain the same 16-state horizon and six outputs.

Training reuses preserved collision-checked demonstrations and motion
augmentation. Goal joints are used only to derive supervised Cartesian target
labels through FK; they are absent from the model observation. The obsolete
goal-joint conditioning weights are zeroed before warm-starting from v2.
Training uses Gaussian noise MSE and validation EMA selection. Original
episode splits and augmented-parent splits stay isolated; test rollouts do not
select training checkpoints.

`train_cartesian_refine.py` optionally adds differentiable FK supervision of
reconstructed clean expert actions to the noise loss. It matches the actual
future demonstration poses, including moving trajectories, rather than
forcing every action to the requested endpoint. Only diffusion noise levels
with cumulative alpha at least 0.25 enter this auxiliary loss. A fixed
validation composite selects the checkpoint. This is training supervision;
runtime still directly samples joint actions.

`compare_cartesian.py` pairs v2 and v3 on the same held-out fixed, moving and
retargeting scenarios. Both use Cartesian candidate ranking, the same full
collision checks and C2 controller, and **neither uses exact joint endpoint
capture**. The v3 environment's IK method is replaced by a raising stub, and
its task dictionary removes goal-joint labels. Success stays at **1 mm and 1°**,
with final errors and additional 5 mm/5° and 10 mm/5° counts reported separately.
Tracking errors use equal simulated durations, including goal holds after
completion. Denoising latency is measured separately from simulated tracking.
Velocity and acceleration remain bounded; jerk is measured, not bounded.

The static controller can truncate a fully collision-checked neural chunk at
its first existing reference within 1 mm and 1°. This captures a predicted
target hit before a later reference overshoots it. It neither creates a joint
endpoint from IK nor reads goal-joint labels. Moving targets retain their full
chunks. Reached static poses use a checked C2 brake/hold of the current pose,
and success requires velocity and acceleration to be at rest.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.diffusion.train_cartesian
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.diffusion.audit_cartesian
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.diffusion.compare_cartesian --only baseline
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.diffusion.compare_cartesian --only candidate
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.diffusion.compare_cartesian
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.diffusion.deploy_cartesian
```

Relative schema 4 uses its own preserved cache, model folder and evaluation
directory:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.diffusion.train_cartesian_relative
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.diffusion.audit_cartesian \
  --cache simulation/artifacts/diffusion_cartesian_relative_v4/cache \
  --output output/diffusion/cartesian_relative_v4_dataset_validation.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.diffusion.compare_cartesian \
  --only baseline --candidate simulation/artifacts/diffusion_cartesian_relative_v4/model/diffusion_policy_6dof.pth \
  --output output/diffusion/cartesian_relative_v4/comparison.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.diffusion.compare_cartesian \
  --only candidate --candidate simulation/artifacts/diffusion_cartesian_relative_v4/model/diffusion_policy_6dof.pth \
  --output output/diffusion/cartesian_relative_v4/comparison.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.diffusion.compare_cartesian \
  --candidate simulation/artifacts/diffusion_cartesian_relative_v4/model/diffusion_policy_6dof.pth \
  --output output/diffusion/cartesian_relative_v4/comparison.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.diffusion.deploy_cartesian \
  --folder simulation/artifacts/diffusion_cartesian_relative_v4 \
  --comparison output/diffusion/cartesian_relative_v4/comparison.json \
  --audit output/diffusion/cartesian_relative_v4_dataset_validation.json
```

The default pilot contains 16 fixed tasks (four blocked routes, four direct
block routes and eight absent-block tasks) and eight moving/retargeting tasks.
`--full` expands to the existing 64 fixed and 24 motion tasks. Fixed tasks run
for at most 100 decisions (40 s), and motion tasks for 120 (48 s). Supply a
different `--output` path to preserve pilot reports when expanding the suite.

`deploy_cartesian.py` creates only a pinned candidate `webui.json`. It verifies
the audit, training, dataset, schema and comparison hashes, strips legacy
`goal_deg` example hints, and records whether all deployment checks passed.
It leaves the active deployment pointer unchanged, including when a candidate
is unqualified. Preserved v2 models and manifests remain available.

Runtime IK independence does not guarantee that an arbitrary Cartesian target
is feasible. Held-out targets are reachable expert labels; unreachable or
blocked requests must remain safe failures. Results describe kinematic
simulation, with no hardware validation.

The initial endpoint-only controller diagnostic is preserved in
[`cartesian_raw_controller/cartesian_v3_comparison.json`](../../output/diffusion/cartesian_raw_controller/cartesian_v3_comparison.json).
Its pilot paired 16 fixed and eight moving/retargeting tasks; the trained v3
policy was evaluated with runtime IK disabled and no goal-joint inputs:

| Initial controller measurement | v2 joint-goal baseline | v3 Cartesian candidate |
|---|---:|---:|
| Fixed strict 1 mm/1° successes | 15/16 | 0/16 |
| Motion strict 1 mm/1° successes | 8/8 | 0/8 |
| Fixed mean final position error | 3.68 mm | 64.11 mm |
| Motion mean final position error | 0.80 mm | 52.04 mm |
| Executed collisions | 0 | 0 |

That candidate failed the deployment accuracy checks. The unqualified
candidate manifest records this failure; it does not activate a deployment.
Safety gates prevented executed collisions, but safe rejection alone does not
establish useful target tracking.

The final relative schema-4 candidate completed another 20,000 training steps
with the FK auxiliary loss, selecting step 20,000 by validation composite
0.00307275 (noise MSE 0.00172331; gated FK loss 0.06747176, weighted by 0.02).
Its cache preserves 120,000 episodes: 95,916 training, 11,992 validation and
12,092 test. The encoder exposes no goal-joint inputs. The frozen model was
then evaluated once on the paired pilot under the shared neural-prefix C2
controller, with the candidate's IK method disabled:

| Final pilot measurement | v2 joint-goal baseline | Relative Cartesian v4 |
|---|---:|---:|
| Fixed strict 1 mm/1° successes | 15/16 | 0/16 |
| Motion strict 1 mm/1° successes | 8/8 | 1/8 |
| Fixed final within 5 mm/5° | 15/16 | 0/16 |
| Motion final within 5 mm/5° | 8/8 | 1/8 |
| Fixed mean final position error | 3.70 mm | 61.05 mm |
| Motion mean final position error | 0.64 mm | 14.64 mm |
| Fixed mean final axis error | 1.32° | 17.94° |
| Motion mean final axis error | 0.34° | 10.01° |
| Fixed tracking RMSE over 40 s | 63.96 mm | 114.36 mm |
| Motion tracking RMSE over 48 s | 39.81 mm | 63.60 mm |
| Executed collisions | 0 | 0 |

The v4 candidate remained below the deployment accuracy requirements. No
larger test expansion or activation was performed. It demonstrates an
IK-independent input and inference path, with useful accuracy still unresolved.
Velocity stayed below 0.987 rad/s and acceleration below 2.982 rad/s² across
candidate rollouts; C2 boundary residuals stayed below 4.7e-13. These safety
results include independent rejection gates and do not imply that raw learned
proposals were collision-free.

The final paired evidence is
[`cartesian_relative_v4/comparison.json`](../../output/diffusion/cartesian_relative_v4/comparison.json).
The candidate checkpoint and training record are preserved in
[`diffusion_cartesian_relative_v4/model`](../artifacts/diffusion_cartesian_relative_v4/model/training.json),
and its unqualified pinned manifest is
[`diffusion_cartesian_relative_v4/webui.json`](../artifacts/diffusion_cartesian_relative_v4/webui.json).

## Experimental visible endpoint selection

`visibility.py` provides `VisibilityEnv`: joint states must pass the
existing collision checks, point-object camera projection, and a URDF line-of-sight
ray test. Inherited edge and smoothing checks therefore include visibility.
Intermediate positions may leave the approach cone. The default camera is a
synthetic 80° × 80° pinhole at the flange, with aligned optical Z; it is not a
measured camera model and does not test the object's full silhouette.

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/compare_visible_endpoints.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/compare_visible_endpoints.py --report-only
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest simulation.tests.test_visibility -v
```

The pilot compares first collision-free, first visible, and minimum-joint-travel
visible routes. Each case/seed shares one endpoint pool: the existing eight-point
disk ordering, a learned-assisted IK pass, and an alternate numerical-start pass.
The common caps are 64 primary IK attempts, 32 fallback attempts, and 800 RRT*
iterations (at most 200 per endpoint). All final routes are replayed at a maximum
0.005 rad per-joint interpolation step. Goal witness joints certify the test
target's reachability but are never supplied to the planner.

The report-only command rebuilds the summary/chart without rerunning the planner.
Chart rendering uses Matplotlib in the active interpreter, or the system Python
when Matplotlib is absent from the robot virtual environment.

Results, timings, source hashes, and selected `.npz` trajectories are written to
`output/visible_endpoint_comparison_20261012`. First-visible and best-visible use
the same checked route bank: their success rates necessarily match, while travel
and consumed search effort can differ. First-visible timing uses the measured
route-bank prefix; both modes include the full shared IK-pool cost. The collision
reference reproduces the first-feasible selection policy, not the production
service's streaming IK or short execution horizon. Timings include diagnostic
visibility replay and must not be interpreted as live controller latency.

These benchmark scripts run offline. The shared controller integration is
described below; hardware qualification still requires measured camera
calibration, full object visibility criteria, and physical feedback validation.

### Bounded improvement after the first visible route

`endpoint_selection.py` retains a fully validated route, then evaluates remaining
endpoints in ascending Euclidean joint-distance order. That distance is a lower
bound on polyline joint travel, so endpoints whose bound cannot beat the saved
route are skipped. The default improvement budget is 350 ms and three extra
attempts, with at most 80 RRT* iterations per improvement attempt and 800 charged
iterations across the search. Before the first solution, the original endpoint
ordering and 200-iteration per-goal limit are retained.

The deadline is checked at collision/visibility samples. A timeout discards the
incomplete candidate and returns the saved checked route. Native checks and final
audit work can cause a small overrun, recorded in the result; this is not a hard
real-time controller. The limit starts after the first valid route, so it does
not bound IK-pool generation or time to first solution. All selected routes still
pass smoothing and dense 0.005 rad collision/visibility validation.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/compare_endpoint_early_stop.py
python3 scripts/plot_endpoint_early_stop.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest simulation.tests.test_endpoint_selection simulation.tests.test_visibility -v
```

The comparison generates a fresh shared IK bank for each case/seed, measures an
independent bounded search, and compares it with first-visible and full-search
policies. It repeats the eight previous scenarios with seeds 42/43 and adds four
held-out direct-visible controls with seeds 44/45. Results are stored under
`output/endpoint_early_stop_20261012`, with the two groups reported separately.
The stopping policy is fixed before evaluation. Matching full search means
matching that finite endpoint/route bank, not global motion optimality. This
eager selector remains the offline reference; the controller uses the streaming
candidate-limited selector described below.

### Generate IK candidates on demand

`streaming_ik.py` interleaves numerical IK and route checks. It uses the same disk
point order, learned seed pass, alternate numerical starts, and 64/32 primary/
fallback IK caps as the eager experiment. Every generated valid endpoint is
considered immediately, so a route can be available before the full pool exists.

The first fully checked visible route starts the shared 350 ms improvement
deadline. It covers both further IK generation and route validation, with at most
three extra route attempts. Known endpoints whose joint-distance bound cannot
improve the incumbent can skip route search; no bound is asserted for ungenerated
IK endpoints. Exhaustion or interruption preserves the last validated route.

The optional `cancel_check` callback on numerical IK propagates through DLS
iterations, line-search steps, and least-squares residual/Jacobian evaluations.
It raises to cancel unfinished work. Existing callers leave it unset. Native
queries and final audit operations remain cooperative rather than hard real-time.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/compare_streaming_ik.py
python3 scripts/plot_streaming_ik.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest simulation.tests.test_streaming_ik -v
```

The comparison reruns the same 24 case/seed pairs, alternates policy execution
order, and checks that completed streamed IK outputs match the eager candidate
prefix. Both policies run fresh IK, and timings include that work. The saved
streaming-first result is a measured prefix, not another independent execution.
Results and trajectories are under `output/streaming_ik_20261012`. These are
offline synthetic-camera results for the original streaming policy, preceding
the candidate-budget changes and controller integration described below.

The 2026-10-12 streaming pilot retained visible routes in all 24 trials. Median
time to the first route fell from 1.738 s to 0.093 s, and median total planning
from 1.793 s to 0.443 s. However, eight routes were longer than eager bounded
selection; the worst increased from 0.338 rad to 2.629 rad (7.79×). In that case,
one difficult IK job consumed the entire improvement window before a better disk
point was generated. This is a material quality regression, despite 16 matching
route costs. See `output/streaming_ik_20261012/summary.json` and
`route_quality_regressions.json`. The follow-up below adds per-candidate limits;
these offline results do not justify replacing the live planner.

### Share the improvement budget between candidates

`CandidateBudget` optionally gives each additional IK job 25 ms and each route
attempt up to 200 ms. Both are clipped to the original 350 ms deadline after the
first fully verified route. The three-extra-route-attempt limit and shared
800-iteration RRT budget remain in force. Interrupted route attempts conservatively
charge their full iteration allowance.

After the first route, a collection window of up to 140 ms gathers additional IK
endpoints under their individual limits. Completed endpoints are ordered by their
joint-distance lower bound before route checks. This prevents an early expensive
route from blocking generation of a later shorter endpoint. If that group is
exhausted while time remains, another bounded collection can run. The collection
window is part of the same 350 ms allowance, not extra time, and is unrelated to
planning across multiple cone slices. `lookahead_seconds=0` disables collection
for the per-candidate-limit-only ablation.

A local timeout discards that candidate's unfinished work and proceeds to the
next candidate. Expiry of the overall deadline returns the validated incumbent.
Completed route results arriving after their allotted deadline are also discarded.
Neither timeout is evidence that the endpoint is unreachable. Numerical solves
are not resumed, and skipped endpoints cannot be certified worse than the selected
endpoint. The source records interrupted IK jobs; selector diagnostics distinguish
local and overall timeouts.

Enable this policy with `choose_streaming_route(...,
candidate_budget=CandidateBudget())`. Leaving the argument unset preserves the
original streaming schedule. Before a first solution, the existing finite
IK/RRT effort limits apply without these wall-clock slices. This protects the
initial search for difficult branches, but does not bound time to first solution
to 350 ms. Cancellation remains cooperative, not hard real-time.

The benchmark independently reruns eager bounded selection, original streaming,
and the candidate-limited policy, rotating execution order and clearing collision
and visibility caches. Its pilot used the 12 known scenarios with one seed each.
A 75 ms route limit was too short to finish checking useful routes. The first
validation used a 200 ms limit and six new controls; one new case with two seeds
exposed route checks starving later endpoint generation. That motivated the
140 ms collection window. Those controls became known cases for the final
evaluation, which adds another six controls with new seeds 48 and 49. Each round
is preserved separately, including the first round's failures. The new cases
extend the same FK generator and have known direct visible witness routes; they
are not a representative deployment test.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/compare_candidate_budgets.py \
  --known-cases output/fair_candidate_budget_20261012/validation/cases.json \
  --new-seeds 48 49 \
  --output output/fair_candidate_validation_new
python3 scripts/plot_candidate_budgets.py output/fair_candidate_validation_new
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest \
  simulation.tests.test_candidate_budget simulation.tests.test_streaming_ik -v
```

Results, trajectories, source hashes and fresh-environment replay checks are
saved under `output/fair_candidate_budget_20261012`. The comparison refuses to
overwrite an existing result set. These results measure the offline selector;
controller integration and its separate validation are described below.

The final 2026-10-12 evaluation (`final_validation/`) completed 48 trials:
36 known case/seed pairs and 12 pairs from six unseen controls. All three policies
found visible routes in all 48 trials. The candidate-limited collection policy
matched eager bounded selection's joint travel in 48/48, improved on original
streaming in 14/48, and was never longer in this sample. Median total planning
was 0.436 s versus 0.439 s for original streaming and 1.726 s for eager selection;
median first-route time was 0.088 s versus 0.089 s and 1.671 s respectively.
Matching this finite reference is not a global optimality guarantee.

All 144 saved trajectories passed fresh-environment replay at 0.005 rad sampling
(41,856 replay samples), including joint limits, start/endpoint acceptance,
collision and modeled visibility checks. Seventy relevant unit/regression tests
passed (`tests_final.log`). The largest observed overall deadline overrun was
7.3 ms, reinforcing that these cooperative limits are not real-time guarantees.
Source snapshots, first-route parity checks and per-trial route improvements are
saved with the final results. The point-target camera assumptions still require
replacement and validation before hardware use.

### Measured-feedback controller integration

`classical_tracking.plan` routes cone-region requests through `region_tracking.py`.
The exact-pose planner retains its existing path. Region planning uses the same
25 ms IK, 200 ms route, 140 ms collection and shared 350 ms improvement budgets
described above. These limits begin after the first fully verified route; they
do not impose a wall-clock bound on finding that first route.

An endpoint must satisfy the frozen stage region, tool pointing and collision
checks. The complete geometric route and its timed continuation must retain
modeled visibility at 0.005 rad sampling. Selection minimizes full geometric
route joint travel; a short executable prefix is never used as the lower bound
for reaching the endpoint. Timing preserves the requested measured q/v/a
boundary and checks the final rest state. An initially invisible object yields
`object_not_visible` with no playable trajectory.

The shared `V2Mouse` controller installs a private encoder-validation callback.
Rounded encoder paths must pass scene, visibility and endpoint acceptance checks
before a candidate can become the incumbent. A rejection allows another endpoint
to be tried. Cooperative deadline expiry during encoder validation discards the
unfinished attempt, preserving any earlier verified route. The public motion API
cannot supply the callback or override the server-owned visibility object.

The controller independently checks the returned curve, any final short-segment
retiming, and the quantized encoder path. During execution it checks measured
joints and each measured-to-command connection against the same visibility
scene. Rejected retargets restore the prior object and retain the checked active
path. Final retiming occurs after candidate selection; a failure at that gate
rejects the proposal without searching additional candidates.

Direct and HTTP followers retain their existing owner and freshness gates:
stale completed proposals are discarded, LOST pauses while keeping the owner and
armed state, and reacquisition replans from measured joints. Each approach stage
captures a measured EEF-to-object axis and retains it across retries. The cone
disk/slab constrains accepted endpoints; intermediate positions may leave it.

Validation logs are in `output/controller_candidate_integration_20261012/`.
They cover real planning with fake motors, independent curve/encoder/live
visibility gates, encoder deadlines, moving joins, approach progression, LOST,
retargeting, and direct/HTTP lifecycle behavior. This is source integration,
not a deployed or hardware-qualified controller. Sampled checks are not a
continuous collision or visibility proof.

### Configured optical mount and full staged simulation

`simulation/camera_model.py` reads the existing ROS flange-to-optical transform.
The current configuration is zero translation and quaternion `[0,0,0,1]`, with
`mount_assumed: true` and `mount_calibrated: false`. Each visibility sample uses
`T_base_optical = T_base_flange(q) @ T_flange_optical` for both projection and the
occlusion ray origin. The view remains a synthetic 80° × 80° field of view, the
object is represented by its center point, and the ray starts 5 mm toward it.
Using the configured transform does not turn these assumptions into calibration.

Direct Python and embedded C++ ROS runtimes pass their mount settings to the
controller. An HTTP controller reads the shared YAML on session startup. The ROS
follower verifies the controller's camera configuration before arming and closes
the new session on mismatch. A session pins an immutable camera model for the
planner, candidate encoder gate, final curve/encoder checks and live executor;
retargets cannot replace it. Offline region planning reads the same configured
defaults. The benchmark `VisibilityEnv()` default remains the identity mount.

The full fake-motor approach test completed all 11 stages from 350 to 150 mm in
20 mm steps at SDK speed 30 and reference speed 0.35 rad/s (controller-capped to
20°/s). It required 13 plans, including checked prefixes and feedback replanning
at 310 and 190 mm. Fresh-scene replay passed all geometric and timed paths at
0.005 rad sampling: 6,422 samples. The final measured fake-motor pose was within
the existing 10 mm axial / 3° pointing arrival allowance: 7.158 mm axial error
and 2.963° pointing error. The smallest sampled center-point FOV margin was
0.196°; this is not a full-object silhouette visibility result.

Evidence is under `output/current_camera_transform_20261012/`: test logs,
`approach_artifacts/test_real_visible_planner_comp0/approach_trace.json` and
`fresh_scene_audit.json`. Replay a saved trace without hardware access:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_visible_approach.py \
  output/current_camera_transform_20261012/approach_artifacts/test_real_visible_planner_comp0/approach_trace.json \
  --output output/visible_approach_replay_new.json
```

ROS and native C++ integration tests use isolated ROS domains, simulated feedback
and dry-run motion settings. Physical motor behavior and actual camera coverage
remain unverified; no physical motion or deployment is implied by these results.

## References

The implementation follows the action diffusion and receding-horizon idea in
[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/). Scheduler usage
follows the official [DDPMScheduler API](https://huggingface.co/docs/diffusers/en/api/schedulers/ddpm).
The [PyBullet guide](https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstart_guide/PyBulletQuickstartGuide.md.html)
documents URDF loading, link-frame FK and closest-point queries. The policy and
RRT* code here are project-specific implementations.
