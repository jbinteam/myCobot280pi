# Five-constraint flange pose controller

The arm has six joints. This task has five independent constraints:

1. Flange X, Y and Z position in the **URDF base frame**, millimeters.
2. A directed unit flange **Z-axis vector in the base frame**, with two degrees of freedom.

Rotation around that directed local Z axis is free. The solver accepts any
flange X/Y orientation consistent with the requested Z direction. At a regular
pose the task Jacobian has rank five, leaving one redundant joint motion.

## Orientation convention

The API accepts `target_axis: [ux, uy, uz]`, normalizes a finite nonzero vector,
and rejects zero/nonfinite vectors. This explicit vector takes precedence over
legacy reference tilts. `u_target = R_target @ [0,0,1]` and `u_actual = FK(q)[:3,2]`
refer to the same `joint6_flange` TCP in `g_base`.

The retained two angle inputs are a **reference tilt**, not Euler angles extracted from the
current flange's full rotation matrix:

```text
R_reference = Ry(orientation_y) Rx(orientation_x)
n_target = R_reference [0, 0, 1]
accepted flange rotations = R_reference Rz(spin), for any spin
```

Angles are right handed in the URDF base convention. The free spin is around
**the flange's own Z axis**. This differs from unconstrained Euler yaw about the
fixed base Z axis, which would generally change the flange pointing direction.
The UI states this convention and shows an orange arrow for the target axis.

Examples:

| Orientation X | Orientation Y | Desired flange Z direction |
|---:|---:|---|
| 0° | 0° | Base +Z (up) |
| 0° | 180° | Base −Z (down) |
| 90° | 0° | Base −Y |
| 0° | 90° | Base +X |

These direction examples do not establish that every XYZ/direction combination
is reachable within joint limits. The target is still `joint6_flange`, with no
tool offset. Position and orientation are derived from the same displayed URDF.

## Web UI

Choose **5D pose · XYZ + tool Z direction** under Target task. It is the default
for a fresh browser session. Enter XYZ, a base-frame direction vector or reference tilts, position tolerance and
direction tolerance. Four classical controllers are available: pseudoinverse,
DLS, DLS with null-space posture centering, and bounded DLS. **Compare all 4** uses the
same start, target, speed cap and budget.

The display has a colored EEF frame, an orange desired Z-direction arrow,
position and direction error plots with separate units, and result/benchmark
columns for both errors. Success requires both errors within tolerance.

**XYZ position only** retains all eight earlier methods and their separate
benchmark. The five trained neural networks only support that position task;
they are not used as five-constraint pose controllers. A separate twelve-input
5D Seed NN supplies four bounded endpoint proposals to PRM; see [SEED5.md](SEED5.md).
PRM orientation is constrained at goals only. The task selection is
remembered in browser local storage.

Every held-out case now includes a reachable direction taken from the same
FK goal that generated its XYZ. Its predefined start applies to simulation.
Hardware planning continues to read the actual robot start, irrespective of
editor poses or held-out starts.
Run and Compare also use the latest live joint telemetry whenever connected,
even after preview edits or playback. Each run captures a new start shared by
all compared methods. Missing/stale telemetry or a moving robot blocks the run.
Disconnected simulation retains the editor/case starting pose.

## Real arm

Start `./start_webui.sh --hardware`, Connect, then **Use live flange pose** to
copy both XYZ and the current pointing tilt. Make a small change, plan from
live joints, inspect the full path, verify the frame/clearance and explicitly
execute. All phase 2 speed, start-drift, one-use plan, expiry, Stop, lease and
telemetry checks remain active. See `../webui/PHASE2.md`.

The plan ID binds the task, position and orientation target on the server.
Orientation/task edits invalidate the browser plan. Targets/settings are locked
during hardware motion; Stop before entering a new goal. Non-convergence in either
position or direction prevents execution. Execution samples include live
estimated direction error. Completion requires joint arrival, estimated
position error ≤5 mm and estimated direction error ≤3°. The default planning
thresholds are 1 mm and 1°; hardware planning direction tolerance is capped at
3°. Free spin itself does not affect success.

All Cartesian measurements are **estimated from joint telemetry and URDF FK**.
There is no external calibration, collision checking, dynamic model, tool offset
or independent orientation sensor. This implementation was verified with fake
hardware; physical performance must be checked on the installed arm.

## Solver

`kinematics.py` now exposes the geometric spatial Jacobian `[Jv; Jw]` directly
from the existing URDF chain. Existing position FK/Jacobian behavior is preserved.

For current flange Z direction `n`, an orthonormal tangent basis `B` has two rows
perpendicular to `n`. The five-row task Jacobian and error are:

```text
J5 = [Jv / 0.3; 0.6 B Jw]
e5 = [(target_position - current_position) / 0.3; 0.6 B rotation_error]
```

The orientation error is the shortest rotation taking current Z to target Z,
using `atan2(norm(cross), dot)`. A deterministic tangent branch handles exactly
opposite directions; anti-parallel axes are a 180° error, not a false zero.
The basis removes angular velocity parallel to local Z, which is unconstrained.
DLS solves the five-row system; pseudoinverse and exact null-space projection
use the same rows. Pure DLS preserves joint 6 spin where possible. The posture
variant can use free motion to center joints. At regular poses the null space
for this URDF is almost entirely joint 6 spin: the first five joints determine
XYZ and Z direction. Thus centering joint 6 has little effect on task accuracy.
Extra null directions occur at singularities, but increasing posture gain does
not generally resolve the task failures.

Integration, joint velocity norm cap, URDF/SDK bounds, gain and budget match the
existing position controllers. Bounded DLS instead solves the regularized
velocity least-squares problem with box constraints from joint position limits
and per-joint speed bounds, then applies the common full-vector speed cap. An
active-set solve redistributes motion when a joint cannot move further. Up to
ten halvings reject nonlinear steps that increase the weighted task-error norm.
It keeps the same damping, gain, speed and rollout budget.

Joint-limit clipping and local Jacobian solvers
can stall on reachable targets; there is no global branch-search guarantee.

## Measured benchmark and checks

Saved `artifacts/benchmark_pose5.json`: 24 reachable poses, seed 9001, default
240 × 0.05 s budget, 1.2 rad/s joint norm cap, 1 mm / 1° tolerance.

| Method | Reached both constraints | Mean XYZ error | Mean direction error |
|---|---:|---:|---:|
| Pseudoinverse | 12/24 | 77.87 mm | 31.46° |
| DLS | 16/24 | 17.56 mm | 8.44° |
| DLS + null-space posture | 16/24 | 17.56 mm | 8.44° |
| Bounded DLS | 19/24 | 25.47 mm | 2.65° |

With the verified operating limits (see `README.md`), bounded DLS improves
success and direction error on this development set, while mean XYZ error is
higher. A separate 96-case set (seed 20260930) reached 64/96 versus DLS 63/96,
with mean XYZ error 27.45 versus 20.35 mm and direction error 3.29° versus 10.63°.
Targets were regenerated within the corrected bounds; these figures cannot be
compared directly with the old limit-profile benchmark. This remains a modest
convergence improvement, not global IK or physical-arm validation. Bounded
solving and extra FK evaluations cost more CPU time per step. Raw outcomes and
timings are in `artifacts/benchmark_pose5_bounded.json`; reproduce with
`OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.benchmark_pose5_comparison`.

These results are separate from the earlier XYZ-only neural benchmark.
Failures are included. The initial UI example reached 0.99 mm / 0.08° with DLS.

Tests cover numerical spatial/direction Jacobians, five-row rank, free spin,
180° orientation error, combined convergence, speed/joint bounds, position-only
network rejection, live planning and orientation-aware completion. Browser
checks verify the target arrow against the displayed flange, orientation
payloads, edit invalidation, error plots and layout without physical commands.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.pose5
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s simulation/tests -v
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s webui/tests_python -v
```

## Solvable workspace from the current start

Open **Reachable and solvable workspaces** and click **Generate workspace
clouds**. Tick **FK reachable cloud · amber** and/or **Solvable cloud · blue**
to choose which clouds to display. Both estimates share the same geometric
candidate pool and requested direction. The transparent-volume checkbox shows
the shells for the selected clouds; the FK shell has lower opacity. Each
checkbox shows its point count. Solver points use larger blue markers that
remain the same screen size as you zoom. A zero-count solvable cloud is labeled
explicitly; FK reachability does not imply convergence from the current start. The server captures live joints if connected and stopped; otherwise
it uses the simulation starting pose (not the playback frame). Every candidate
is tested with the production solver from that same fixed start, using the
selected method, step budget, speed, damping, position tolerance, requested
flange direction and direction tolerance. Only converged target poses appear
as blue points. Metadata records the start and individual success/failure results.

FK generates geometric candidate positions within operating joint limits.
For pose5, local direction projection supplies candidates with the requested
flange Z direction. This generation is biased and incomplete; FK candidates
are not automatically admitted to the displayed cloud. The actual selected
solver must converge from the captured start. The current flange position is
also tested against the requested direction. XYZ-only mode leaves direction free.

Choose 64–512 targets to test and a candidate pool size. Larger target counts
cost more solver rollouts. The cloud shows only the selected direction slice
of the pose workspace, not all orientations at every XYZ. Solver failures and
empty regions are not proofs of physical unreachability. No collision or
physical-path validation is performed.

The transparent voxel surface approximates an envelope around successful
sampled targets. Its neighboring points are not individually solver-tested.
Cells are retained when their center is within position tolerance plus half a
voxel diagonal of a tested target; the shell can extend at most one voxel
diagonal beyond these balls (13.9 mm for 8 mm voxels). Internal faces are removed,
preserving gaps rather than filling a convex hull. Adjust resolution, visibility
and opacity as needed.

Previous clouds remain visible after held-out case, starting pose, direction,
solver or tolerance changes, with a message to regenerate. Live pose changes
larger than 1 degree trigger that message; estimation itself uses exact captured
joint telemetry. Camera/playback changes do not change the simulation start.
EEF offsets resolve targets and do not translate the flange workspace.

Export the same solver-tested cloud and voxels as JSON:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.workspace \
  --start-deg 0 -40 -130 80 0 50 --method dls --targets 128 \
  --task pose5 --orientation 0 180 --direction-tolerance 1 \
  --tolerance-mm 1 --samples 12000 --voxel-mm 8 \
  --steps 240 --speed-rad-s 1.2 --damping 0.04 \
  --output simulation/artifacts/workspace.json
```

The viewer requests `/api/sim/workspace`; restart an older bridge after updating
code. The estimate performs calculations and telemetry reads, with no motion.
