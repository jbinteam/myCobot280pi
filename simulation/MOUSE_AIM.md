# Full 5D mouse aim (simulation only)

## Cone and disk tracking

The WebUI's **Learned IK + classical RRT*** and **Diffusion v2** tracking engines
use a circular acceptance region. **Distance to circle** defaults to **100 mm**
and **Cone half-angle** defaults to **20°**, giving a radius of approximately
**36.4 mm**. The cone apex is the cursor. For each cursor target, capture the
unit vector `u` from the EEF toward the cursor and hold it fixed during planning
and execution:

```
center = cursor - distance * u
radius = distance * tan(half_angle)
```

The disk plane is perpendicular to `u`. An acceptable endpoint lies inside its
radius and within the requested position tolerance of its plane. Its local
tool Z must point from that endpoint toward the cursor within the requested
direction tolerance; local Z spin remains free. An off-center endpoint can be
farther than 100 mm from the cursor because 100 mm is the axial distance to the
plane. The acceptance region applies to the endpoint; the entire approach path
still receives the existing joint-limit, self/table/block and timing checks.

The planner tries bounded disk candidates, beginning with the point closest to
the current EEF, and reports the selected endpoint separately from the disk
center. A finite candidate search can fail even when another disk point might
be reachable. The viewport shows the cone and disk, plus a single selected
final target marker and the shared joint velocity profile.

Cone planners report `tracking_objective: gaze_first`, per-frame cursor-facing
error and aggregate gaze diagnostics. Diffusion fallback ranks final and mean
gaze error before disk displacement and joint travel. Gaze is checked at the
endpoint; there is no guarantee of constant pointing through a collision detour.
Hardware uses feedback-paced progress and replans from live joints, while the
simulation retains its checked continuous-curve playback.

Requests use `tracking_region: {mode: "cone_disk", distance_mm: 100,
half_angle_deg: 20}` and `target_axis` for the frozen cone axis. Responses include
canonical `tracking_region.axis`, `center_mm` and `radius_mm`; response
`target_mm` and `target_axis` describe the selected EEF endpoint and its pointing
direction. Requests without `tracking_region` retain the legacy exact 50 mm
point target described below. Classical IK tracking retains that legacy mode.

This page describes the default simulation mode. With `--hardware`, the same
viewport uses a separate live-start hardware controller; see
[hardware mouse aim](../webui/HARDWARE_MOUSE_AIM.md).

Start the UI with `./start_webui.sh`, open http://127.0.0.1:8000 and click
**Start mouse aim** above the 3D scene. Move the mouse inside the scene. Camera
orbit pauses tracking; move inside the scene again to resume. Escape, Stop, manual
pose edits, experiment actions, and settings edits stop tracking. The simulation
endpoint requires hardware to be disabled and disconnected; it never prepares
or executes a hardware plan and never reads robot telemetry. Existing XYZ mode and checkpoints remain
available for the separate experiment controls.

## Coordinates and target

The cursor is the intersection of the mouse ray with a camera-facing plane,
anchored 50 mm along the starting flange Z axis. The plane remains anchored and
faces the current camera. The browser sends `cursor_mm` in robot base **g_base**
and exact `start_deg`, without rounding, to `POST /api/sim/mouse-aim`.

For the captured flange position `p_current` in millimeters:

```
u_target = normalize(cursor_mm - p_current)
XYZ_target = cursor_mm - 50 * u_target
```

FK uses the same `joint6_flange` TCP with zero local offset as the viewer and 5D
model. The derived target is already in the base frame, so this endpoint sends
zero local offset to the existing planner. The ordinary planner's convention,
`base_target + R_start @ eef_local_offset`, is preserved. Tool direction is
`R_flange @ [0,0,1]`, expressed in g_base. This controls three position degrees of
freedom and two direction degrees of freedom, with rotation about **local** tool Z
free. Opposite directions differ by 180 degrees. Orientation is constrained at the
goal only; paths may temporarily change tool direction and distance from the mouse.

## Local tracking first

Each cursor update first tries the selected local 5D IK controller from the
captured current joints. In XYZ experiment mode this remains 5D DLS. The complete
solver path must pass operating joint limits and 1 mm self-collision clearance,
including every intermediate state and segment at 0.04 rad per-joint sampling.
The endpoint must pass both the nominal pose tolerances and actual cursor gap
and direction acceptance.

Successful local paths are simplified only through freshly checked shortcuts,
then retimed and checked again for speed-limited playback. This avoids making
each small solver update a mandatory hardware stop. A successful local update
does not load the roadmap or run neural goal proposals. Simulation and hardware
share this policy; hardware captures its starting joints from fresh telemetry.
The UI identifies local IK from current joints. API results expose
`tracking_strategy: local_first`, `tracking_source`, and `local_ik` attempt
diagnostics.

A local solve that stalls, exhausts its budget, misses the goal acceptance or
produces an invalid path falls back to the existing pipeline below. No partial
local prefix is played on failure.

## Optional local IK → diffusion tracking

Above the scene, select **Local IK → diffusion · self / table / block**, then
**Start mouse aim**. Set the block using the diffusion panel; it is shown with
the table throughout hybrid tracking. The original Seed NN / PRM engine remains
the default, and hardware mouse control uses its existing engine.

The hybrid engine first solves from the captured current joints and checks the
complete local path against the same self/table/block scene as diffusion. If
that path fails, a scene-valid 5D endpoint conditions the deployed DDPM. All raw
16-reference proposals, speed-projected paths and connecting segments are
checked before selecting the next two 0.1-second references. Local IK selects
up to the next 0.2 seconds. The selected prefix then receives checked shortcuts
and quintic timing with 0.02-second frames, lasting at most 1.3 seconds and
starting/ending at rest. Every retimed state and edge is checked again.
An accepted prefix is `playable: true`;
`converged` means the executed endpoint actually satisfies cursor acceptance.

Replanning begins from the exact displayed joints after a chunk finishes, even
when the mouse is stationary. A cursor snapshot retains its directed target
axis across chunks; a previous IK goal is only a hint and is checked again for
the current target and scene. The browser discards changed-cursor responses,
holds while computing and checks the returned scene/checkpoint/start, smoothing,
speed and acceleration. Normal retargeting waits for the chunk to finish at rest.
Settings changes, leaving the scene and Escape cancel playback. The selected
decision budget bounds repeated requests for one cursor target; five successive
unsafe-proposal decisions hold until the mouse moves.

This is experimental reuse of a fixed-goal checkpoint, without moving-target
training. Planned references cap vector speed at 1 rad/s and acceleration at
3 rad/s². Immediate cancellation can interrupt a chunk. Large
cursor jumps and some nearby targets can fail or stall. No unsafe or unvalidated
prefix is played. Timing includes server computation; cold scene/model loading
is slower. This mode does not create hardware plans.

Run `.venv/bin/python -m simulation.diffusion.benchmark_tracking` to replay six
cursor scenarios and independently audit every executed state/edge, speed and acceleration.
The default results are written to `output/diffusion/mouse_tracking_benchmark.json`;
the post-smoothing replay is saved in `output/diffusion/smoothed_mouse_tracking_benchmark.json`.

## Seed NN / PRM fallback

The endpoint requires self-collision checks with the existing 1 mm clearance. Its
pipeline choices cannot be disabled through request fields:

1. Validate start/target; the already-at-goal shortcut requires both position and
   directed tool-axis tolerances.
2. Use the trained versioned `seed5_v1/models.json` checkpoint. Check all four raw
   bounded NN guesses with FK and goal acceptance; collect unique valid goals.
3. Independently queue every finite, in-limit NN guess for 5D numerical refinement,
   including accepted and initially self-colliding guesses.
4. If no raw goal passed, append the current configuration, up to four pose-near
   roadmap configurations, and four deterministic random seeds. This fallback
   does not depend on the NN refinement flag. A missing/incompatible 5D checkpoint
   uses the supported numerical fallback and is reported explicitly.
5. Deduplicate seeds, refine in NN/current/roadmap/random order using the selected
   5D controller, with automatic classical DLS and bounded DLS recovery. In XYZ
   experiment mode, mouse aim still uses 5D DLS. Refine until the queue
   is exhausted or four unique valid goals are accepted. DLS iterations are
   computations, never preview motion.
6. If the normal queue finds no goal, run a separate bounded coverage pass:
   up to 32 unique seeds from failed-solver endpoints, original NN/current poses,
   diverse roadmap postures, NN proposals conditioned on other bounded anchors,
   and deterministic stratified configurations. Each trust-region solve allows
   at most 96 optimizer function evaluations, plus derivative/acceptance checks
   and separately recorded antipodal escape probes. All normal and actual
   cursor checks apply.
7. Try valid direct joint-space connections, otherwise attach start/goals to
   the compatible existing PRM and search. Check edges, optional shortcuts,
   final path and speed-limited trajectory. On attachment/routing failure, try
   one coverage pass for additional distinct goals, expanding the goal budget
   to eight, then retry routing if new goals were found. A trajectory-validation
   failure does not restart IK. With **Repair blocked routes** enabled, a
   remaining attachment/connectivity failure permits one 32-node guided
   temporary roadmap repair before final trajectory validation. No waypoint
   IK is needed.
8. Require the actual cursor gap and pointing angle at the shortcut, raw-goal
   acceptance, numerical termination and goal admission, before a goal consumes
   the accepted-goal budget. Check the final trajectory endpoint again. Only a
   successful, self-checked path passing that check is played.

This reuses `simulation.service.plan` and `simulation.planning.query`; checkpoint,
roadmap and joint-space connectivity are reused. The normal seed-queue rules
and four-goal budget remain; coverage and eight-goal routing recovery are
separate bounded stages. Goal sources,
checkpoint status, IK outcome and route outcome are shown independently. The first
fallback request may take substantially longer while the persisted roadmap is
revalidated; subsequent fallback requests reuse the validation cache. Model training and numerical
fallback status are described in [SEED5_RESULTS.md](SEED5_RESULTS.md).
Recovery details, commands and benchmark results are in [IK_RECOVERY.md](IK_RECOVERY.md).
The shared **Repair blocked routes** checkbox is enabled on first use, remembers
its setting, and controls both PRM previews and mouse aim. The mouse-aim panel
shows its result and cost. Changing the checkbox stops active mouse aim;
**Compare this target** retains the captured setting. API callers enable this
with `guided_route_repair: true` and may set `route_repair_samples` (1–128,
default 32). The additional route stage is described in
[GUIDED_REPAIR.md](GUIDED_REPAIR.md).

## Tolerances and playback

The UI's position tolerance (mm), direction tolerance (degrees), speed norm
(rad/s), DLS damping and step budget apply without hidden tightening. Defaults
are 1 mm and 1 degree for the 5D task. A position error also changes the direction
from the endpoint to the cursor, so actual cursor acceptance additionally checks
`abs(distance_to_cursor_mm - 50) <= requested_mm` and
`atan2(norm(cross(actual_Z, direction_to_cursor)), dot(...)) <= requested_deg`.
This extra condition applies to all accepted endpoints, independently of the
nominal XYZ/direction errors. A numerical solver keeps refining when the nominal
task passes but the actual cursor geometry does not. A rejected endpoint does not
consume the goal budget. Failure holds the preview; direction and self-collision
requirements are retained. The readouts distinguish planned goal errors, best
candidate errors on failure, and current cursor errors.

**Compare this target** stops tracking and transfers the exact captured starting
joints, derived XYZ/direction, zero local offset, controller and settings into the
experiment controls. This avoids comparing different targets or rounded starts.
The last completed target stays available while another request is computing and
after the pointer leaves the scene. An aborted or late response cannot replace
it. Starting a new mouse-aim session clears the saved target. Press Escape before
moving toward the button to preserve a particular result without further tracking.
Standalone XYZ controllers do not constrain pointing; standalone 5D controllers
check their nominal target direction without collision checks. PRM adds collision
and route checks. Mouse aim also checks the actual pointing direction to the cursor.

The browser holds the exact starting preview pose during one in-flight query and
combines cursor changes into the latest target. A valid response to an earlier
cursor snapshot may begin playback while a newer cursor is pending; after at
least 200 ms it replans from the current validated path configuration. This lets
continuous cursor movement make progress instead of rejecting every response.
Changes smaller than 0.5 mm are coalesced. A failed stationary target is held until
the cursor moves, rather than retried indefinitely.

Playback linearly interpolates consecutive checked joint frames, preserving PRM
corners and the configured speed norm. Every checked frame is visited; an animation
update that spans a corner stops there and drops surplus time, slowing playback.
It never repeats IK at known waypoints.
Paused/hidden tabs do not catch up accumulated time. Pause/stop/edit/hardware state
changes invalidate the request generation, abort delivery and clear playback.
The browser rejects mismatched start/cursor, incorrect validation/task/frame tags,
nonfinite/out-of-limit frames, invalid timestamps and excessive speed. The server
allows only one mouse planner computation at a time, including an aborted query
that is still finishing, and returns HTTP 409 for bounded latest-target retry.

## Verification

```
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s simulation/tests -v
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s webui/tests_python -v
cd webui
npx playwright test tests/mouse-aim.spec.js tests/prm.spec.js tests/seed5.spec.js tests/pose5.spec.js tests/simulation.spec.js tests/viewer.spec.js
npm run build
```

New checks cover cursor geometry and finite/limit validation, local-Z spin,
already-at-target, trained NN and missing-model fallback, endpoint cursor
acceptance, full self-checked frames/speed, server hardware gating and busy-lock
release, real browser convergence, cursor coalescing, late-response cancellation,
separate IK/route failures and defensive trajectory gating. Existing core tests
cover all-four-guess handling, seed order/uniqueness/budget, opposite directions,
XYZ checkpoint compatibility and obstructed PRM routing.

The full planner introduces visible computation latency; this is a replanning
preview, not a continuous hardware servo. Collision checks retain the project's
conservative convex URDF assumptions and finite sampling resolution. They exclude
external obstacles, added tools and cables. Finite IK/search failure does not prove
unreachability. No new model training was required for this integration.

Workflow source: [Mermaid](../output/workflow/mycobot_mouse_aim_full.mmd).
Rendered workflow: [PNG](../output/workflow/mycobot_mouse_aim_full.png) /
[SVG](../output/workflow/mycobot_mouse_aim_full.svg).

## Initial integration validation (2026-10-02, before the tolerance fix)

All **149 checks passed**: 82 simulation tests, 45 bridge/fake-hardware tests,
8 mouse-aim browser tests and 14 existing UI regression tests covering XYZ,
5D direction entry, PRM and desktop/mobile rendering. The production build passed.
The running server reports hardware disabled, disconnected and no active motion.

Four simulation queries against the running HTTP server passed both cursor
constraints. The already-at-target case used `current_state`; three changed
cursor targets used the ready trained 5D checkpoint and produced accepted goals
from NN refinement and current-configuration numerical fallback. All four routes
were direct connections; obstructed PRM routing remains covered by core tests.

| Cursor change (mm) | Final gap (mm) | Pointing error (degrees) | Warm HTTP query (ms) |
|---|---:|---:|---:|
| 0, 0, 0 | 50.000 | 0.000 | 51 |
| 4, 3, 0 | 50.096 | 0.410 | 212 |
| -3, 2, 1 | 49.835 | 0.388 | 227 |
| 5, -2, 0 | 50.123 | 0.392 | 218 |

These are four smoke queries, not a workspace-wide mouse-tracking benchmark.
Full results are in [mouse_aim_live_validation.json](../output/mouse_aim_live_validation.json).
The selected trained checkpoint is reused without retraining; SHA-256 checks
confirmed the original XYZ model, weights, dataset, training metadata and
roadmap are unchanged.

Changed implementation files are `simulation/service.py`, `webui/server.py`,
`webui/src/mouse-aim.js`, new `webui/src/mouse-aim-path.js`, `webui/src/main.js`,
`webui/index.html` and `webui/src/style.css`. Checks were added in
`simulation/tests/test_mouse_aim.py`, `webui/tests_python/test_server.py` and
`webui/tests/mouse-aim.spec.js`. Documentation is in this file, `SEED5.md`,
`simulation/README.md` and `webui/README.md`; workflow sources/renders are under
`output/workflow/mycobot_mouse_aim_full.*`, with UI previews under
`webui/preview/mouse-aim-full-{desktop,mobile}.png`.

## Tolerance and solver mismatch fix (2026-10-02)

The first integration privately tightened displayed 1 mm / 1 degree tolerances
into approximately 0.393 mm / 0.5 degree. This was sufficient for cursor accuracy
but unnecessarily rejected some poses that already met the actual requested gap
and angle. It also forced bounded DLS while the numerical experiment used the
selected controller, commonly classical DLS.

A reproduced case starts at `[20,-35,-90,40,10,0]` degrees with cursor
`[285.69709856079663,32.10990138603191,-33.48065176338741]` base millimeters.
Classical 5D DLS finds a collision-valid goal with about 0.999 mm nominal position
error, 0.049 degree nominal direction error, 0.771 mm actual gap error and
0.764 degree actual cursor pointing error. The old hidden position threshold
rejected it even with 600 steps. It now produces a validated direct path. Another
regression check covers a perfectly aimed 50.8 mm gap with a one-step budget:
it now correctly uses the already-at-target shortcut instead of reporting IK failure.

Displayed tolerances now apply directly. Actual cursor geometry is an additional
acceptance condition throughout the IK pipeline, including raw proposals and
numerical termination. Rejected cursor goals cannot consume the four-goal budget.
The selected 5D solver is honored and classical/bounded DLS recovery is available
on each unique seed. Numerical iterates still are not motion, and all final goals
and paths retain the existing collision, limit and trajectory checks. Failure
messages report accurate-but-invalid candidates, cursor rejection and the best
collision-valid errors. **Compare this target** copies exact inputs for a fair
numerical comparison, including full-precision starting joints.

The reproducible 75-case cursor-plane sweep retained all 55 previously accepted
IK tasks and recovered the case above: **56/75 accepted IK tasks and 56 validated
direct paths**, with no lost IK cases. Nineteen searches still found no acceptable
5D goal. In thirteen of those, XYZ-only DLS/NN+DLS reaches a collision-valid
position, but its nominal pointing error is about 28–157 degrees, outside the
mouse-aim direction tolerance. Other comparisons found pose-accurate numerical
endpoints rejected by self-collision. These results describe the sampled cases;
finite search failure does not establish 5D unreachability. No checkpoint or
roadmap was retrained or replaced.

Results: [75-case benchmark](../output/mouse_aim_recovery_benchmark.json) and
[live-server reproduced case](../output/mouse_aim_recovery_live.json).

```
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.benchmark_mouse_aim
```

Additional changed files for this fix: `simulation/planning.py`,
`simulation/pose5.py`, `simulation/benchmark_mouse_aim.py`,
`simulation/fixtures/mouse_aim_recovery_cases.json`,
`webui/src/simulation-controls.js`, `simulation/tests/test_seed5.py` and
`simulation/tests/test_pose5.py`, alongside the mouse-aim files listed above.

Validation after the fix: **157 checks passed** (89 simulation, 45 bridge/HTTP,
9 mouse-aim browser and 14 XYZ/5D/PRM/viewer regression checks). The browser
checks include exact-input comparison and projection of the cursor marker back
to the mouse position. Desktop/mobile previews and the workflow renders were
inspected. The production build passed. SHA-256 checks confirm the five original
XYZ assets and the selected 5D checkpoint/configuration are unchanged.

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s simulation/tests -v
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s webui/tests_python -v
cd webui
npx playwright test tests/mouse-aim.spec.js
npx playwright test tests/prm.spec.js tests/seed5.spec.js tests/viewer.spec.js tests/simulation.spec.js tests/pose5.spec.js
```

The regression browser tests use the simulation-only server on port 8000;
start it with `./start_webui.sh` from the project root. No physical robot motion
was executed during these checks.

Performance measurements, code hotspots and researched optimization options
are documented in [PERFORMANCE.md](PERFORMANCE.md). Reproduce them with
`OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.profile_mouse_aim`.
