# myCobot 280 Pi simulation and web control

A local URDF viewer and pose controller. Five-constraint XYZ + flange Z direction
control is the default target mode; local flange-Z spin is free. See
[five-constraint controller](../simulation/POSE5.md) for the orientation convention. Phase 1 runs eight classical/neural
position controllers entirely in simulation. The server defaults to rejecting
hardware connect/move/jog commands. See [simulation documentation](../simulation/README.md)
for the methods, training, benchmark and limitations.


## Fixed-target default: learned IK + classical RRT*

The WebUI selects **Learned IK + classical RRT*** for fixed targets. It refines
the evaluated eight learned virtual IK starts with the existing 32-start bounded
numerical solver and robust fallback. A checked direct joint path is tried first;
RRT* supplies detours when necessary. Complete paths receive collision-checked
quintic timing before preview or hardware preparation.

Use **Plan learned IK + RRT*** in simulation or **Plan from live joints** for
hardware. The fixed-target selector also retains diffusion v2 and legacy IK.
Mouse tracking also defaults to learned IK + classical RRT*, using continuous
C2 references and the same scene checks. It uses a cone with its apex at the
cursor and an acceptable perpendicular disk, defaulting to 100 mm distance and
20° half-angle. The EEF can reach anywhere inside the disk while its local Z
points toward the cursor. See [tracking geometry](../simulation/MOUSE_AIM.md)
for settings and endpoint acceptance. The 10,000-case fixed-target
comparison had no external blocks; it does not establish moving-target or
physical tracking accuracy.

The new simulation endpoint is `/api/sim/classical`. Hardware uses the existing
checked preparation route `/api/ik/diffusion/plan` with `planner: "learned_rrt"`.
Its checkpoint identity refers to the learned IK weights, not diffusion weights.
Live-start, scene review, route replay, speed, heartbeat, Stop and endpoint
settling checks apply through the existing executor. A changed learned checkpoint
invalidates a reviewed plan. Tests use fake hardware; deployment sends no motion.

## Motion specification checks

Planned UI motion uses the [manufacturer's 280 Pi motion table](https://docs.elephantrobotics.com/docs/mycobot_280_pi_en/3-FunctionsAndApplications/6.developmentGuide/CommunicationProtocolPackage/18-communication.html):
150°/s and 200°/s² per joint, and 100 mm/s and 400 mm/s² per Cartesian axis.
`simulation/motion_limits.py` enforces XYZ limits conservatively as flange
translation norms. It also applies a 40°/s, 66°/s² angular-vector policy;
angular vectors are not Euler-angle derivatives near singular orientations.
The bound uses URDF link translations and derivative control hulls, including
the velocity-squared contribution to Cartesian acceleration between samples.
It covers the bare `joint6_flange`, not an added tool or payload torque.

Ordinary IK/PRM previews retain their route and use rest-to-rest quintics at
every retained vertex. Numerical solver results and training are unchanged;
playback can take longer. Their maximum selectable joint speed norm is 2.6 rad/s.
Diffusion quintics and continuous C2 joins also pass the specification envelope;
an infeasible moving boundary is rejected instead of changing its q/v/a join.

Hardware IK and mouse execution check actual rounded SDK target increments
against monotonic send intervals. Fixed-target and mouse execution can wait
for admissible timing; classical tracking also limits combined flange derivatives.
A final rejected v2 send stops execution. Status/samples expose checked command
velocity, acceleration and conservative bounds. These are send-interval averages,
not a continuous wall-clock C2 guarantee or measured servo dynamics. SDK speed
remains a separate firmware setting. Fixed v2 velocity-profile plans allow 1–80%; bounded encoder trials allow
1–5%. Angle-command controls allow 1–80%. Manual Move/Jog keep their SDK controls.
Fixed-target hardware plans use that percentage to set a per-joint allowance:
`150 degrees/s × selected speed / 100` (120°/s at 80%). Each straight leg uses
signed joint displacements divided by their common maximum and the shared
quintic velocity shape divided by its peak. Multiplying by the allowance gives
coordinated joint velocities; a common time stretch enforces acceleration and
flange limits, so the resulting peak can be lower. Classical and v2 plans retain
the executor's route and stops. The reviewed timing paces waypoint sends, with
telemetry, command audits and settling allowed to slow execution further.
Changing arm speed invalidates the preview and requires replanning. SDK speed
does not guarantee a measured angular speed or firmware interpolation curve.
Physical tracking and measured acceleration require a supervised arm trial;
fake hardware tests cannot establish them.

## Start

From the project root, install frontend dependencies once:

```bash
npm install --prefix webui
```

Then run:

```bash
./start_v2.sh
```

Open http://127.0.0.1:8000. This builds the frontend and starts the local Python
bridge. Requires Node.js 18+, the project `.venv`, and a WebGL-capable browser.
The Pi and its TCP server are not needed for simulation.
The launcher checks the pinned v2 model and starts in **Simulation only**.
Use the **Mode** selector above the workspace to choose **Simulation only** or
**Simulation + hardware** without restarting the server. Local IK + Diffusion v2
and continuous motion stay the defaults in both modes.

Simulation only hides real-arm controls and rejects hardware commands. Click
**Start mouse aim**, move inside the scene, and press **Escape** to stop the preview.
For the automatic approaching task, click **Start approach demo** under
**Approaching simulation**. This loads the tested pose and holds the orange
object fixed. The first disk uses the actual EEF-to-object distance captured
from fresh preview FK, then stages decrease by 20 mm toward 150 mm. The standard
350 mm start still gives 350, 330, …, 150 mm. A start already closer than 150 mm
holds its initial-distance disk after alignment, without requesting an outward
stage. Stage counts and distance readouts follow that measured start. The current
configured wrist-camera transform and visibility checks apply to every route.
Each stage uses fresh preview joints; a partial route or failed search does
not advance the distance. **Pause** and **Resume** reanchor the current stage
from the paused pose. **Stop approach** or **Escape** cancels pending planning
and preview motion. **New random object after completion** is enabled by
default: after the final stage, it waits two seconds, resets the virtual arm
to a matching start pose with a random base heading over a 200° arc, places a
new object with an additional 10–60 mm local variation, and runs the
next approach. The object stays fixed for the entire round. Uncheck this to
hold after one round. Pause/Resume also work between rounds; Stop cancels repeat.

**Allow camera-FOV fallback if strict planning fails** is enabled by default for
simulation approaches. Each request first runs the original strict pointing
planner. Any checked playable result, including a partial motion, is retained.
Only `ik_failed` or `no_path` with no playable motion triggers a second pass with
orientation relaxed inside the configured camera frame. Both passes share the
same starting motion boundary, frozen stage disk and pinned camera. Invalid
starts, visibility failures, exceptions and cancellation do not trigger fallback.
The fallback adds its processing time to the strict attempt; both keep their
normal per-pass search budgets.

The fallback endpoint must reach the same disk and keep the object center inside
the rectangular camera image bounds. Joint limits, collision, occlusion,
complete-route visibility and timed-reference checks remain mandatory. Changing
the option stops the current run; uncheck it for strict-only comparison. The
replacement case at `/?case=ik-failed-reachable-150` uses the same default policy
and receives no saved endpoint or route. The saved-case baseline button remains
strict. Camera calibration and physical qualification remain deferred.

API requests use `orientation_mode: strict_then_fov` for the hybrid policy.
Responses report the effective `orientation_mode` (`point_at_object` or
`camera_fov`) and `orientation_selection` with both pass outcomes and processing
times. The simulation approach endpoint defaults to this policy; ordinary
mouse-aim requests retain strict pointing unless explicitly configured.
Explicit `camera_fov` remains available for the experimental comparison.

**Show RRT* search** displays live 4-pixel magenta branches as the real joint-space tree
grows and rewires. Branches are tool-position projections; their straight
display segments are not commanded Cartesian paths. The 6-pixel blue line shows the
checked timed route. Both use screen-space wide lines with white outlines for
contrast against the robot and background. Direct-path stages are labeled without inventing tree
expansion. The simulation-only `/api/sim/approach-plan` endpoint streams bounded
NDJSON search batches and then the fully checked proposal; tree updates never
move the preview. The demo is available only in disconnected simulation mode
and does not send motor commands.

For the visibility-aware planner, first click **Load visible tracking demo**.
This loads the tested above-table pose, no block and a 350 mm / 30° cone.
After Start, the first pointer entry preserves the target at the wrist camera
centre; gentle mouse movement then steers it relative to that point. It is a
simulation preset and is unavailable in hardware mode. The ordinary default
pose points down: a 350 mm target along that axis lies behind the table.
The wrist camera differs from the orbiting 3D viewport. Targets outside its
configured view or behind scene geometry remain held; the status identifies
the obstruction or view condition. The preset does not bypass these checks.
Simulation + hardware enables arm setup and live-joint simulation previews;
connect, confirm the frame/table and enable movement before starting real-arm
mouse tracking. Selecting this mode does not connect or move the arm.
Returning to Simulation only stops tracking, disconnects the arm, invalidates
pending hardware plans and reloads the workspace with confirmations cleared.
The mode is shared by all tabs; open tabs reload when it changes.

For a server locked to simulation, use `./start_v2_simulation.sh` or
`./start_webui.sh --simulation-only`. Its hardware option is disabled. The
server rejects combining `--simulation-only` with `--hardware`.
`./start_webui.sh` supports selection and starts in simulation; `--hardware`
starts with hardware enabled and still allows returning to simulation.

For frontend development, run these in two terminals from the project root:

```bash
.venv/bin/python webui/server.py
npm run dev --prefix webui
```

Open http://127.0.0.1:5173. Vite forwards `/api` requests to the bridge on port
8000. Both services bind to the PC's loopback address.

## Simulation lab

Custom v2 targets first require a collision-free joint endpoint matching XYZ
and the directed tool-Z axis within 1 mm / 1°. The endpoint search uses
32 bounded-DLS starts total (up to 400 iterations each). The live joint pose
is tried first. The v2 launchers enable a trained network that proposes eight
virtual initial guesses, replacing up to eight random starts after that first
attempt fails. Every guess is numerically refined and checked. If they fail, v2
tries up to 32 deterministic bounded least-squares starts, with at most 160
function evaluations per start. Learned guesses can select different valid
joint solutions. Set `MYCOBOT_IK_SEED=0` before launch for numerical starts only.
The fallback uses the same joint limits, self/table/block collision checks and
final tolerances. It changes endpoint search only; the pinned v2 neural model,
trajectory validation and execution checks remain in effect.
See [learned IK starting guesses](../simulation/diffusion/IK_SEED.md) for the
dataset, training, held-out results, model controls and limitations.

Plan results expose `ik_diagnostics`: solver method, attempt counts, fallback
evaluations, elapsed time, best paired position/direction errors, and collision
rejection calls (which can include repeated candidates). Search exhaustion does
not prove a target is unreachable. Hardware feedback shows the best errors and
suggests adjusting either position or direction. The classical controller's
step-budget field does not configure this v2 endpoint search.

Historical numerical-fallback evidence and a reproducible comparison are in
`output/ik_robustness_20261007/`: 15/16 known-reachable targets solved by the
original search versus 16/16 with the fallback, preserving all 16 previously
successful outputs including the user's adjusted target. This small corpus
checks endpoint robustness, not full-path or physical execution success.

The default screen is the mouse-tracking workspace: a large robot view, model
selection, and (in hardware mode) grouped connection and movement confirmations.
Tracking defaults to **Local IK + Diffusion v2**, continuous C2 motion, no block,
1 mm / 1° planning tolerances, and the bounded DLS pose solver. Hardware arm
speed defaults to 70%, with an 80% ceiling and separate 5% encoder trial caps. The active checkpoint is unchanged; v3/v4 remain unqualified.

Fixed-target planning defaults to **learned IK + classical RRT*** with
**Velocity profile · encoder + joint speeds** (`joint_command: "encoder_drag_profile"`).
When diffusion v2 is selected as the alternative, the pinned v2 model generates the
checked route using the original full-horizon C2 rollout. Collision-checked
final shortcuts retain necessary corners; the selected speed allowance retimes
that route to a quintic q(t). Fixed-target v2 uses this original method again;
the faster waypoint-only rollout has been rolled back.
Collision checks share exact cached states and already-checked endpoints on the
same interpolation grid. Fixed v2 ranks speed-projected candidates before
validating them, stopping once the original selection rule accepts a checked
C2 curve. All inference calls and random draws are retained. Rejection counts
describe considered candidates; lower-ranked unused candidates are not checked.
The driver derives signed dq/dt analytically and sends six instantaneous speed
magnitudes together with route encoder targets using `set_encoders_drag`.
Direction comes from the target. This is a position/speed adapter, not a signed
firmware velocity mode; conversion to steps/s remains nominal.

Telemetry queries begin before each 0.2-second send deadline, using the previous
measured query duration. Early replies wait outside the controller lock so Stop
and heartbeats remain responsive; overruns use actual time without catch-up
bursts. Samples report `telemetry_cycle_s`, `deadline_wait_s` and
`send_deadline_lateness_s` alongside actual send intervals.

Each plan records `planning_timing_ms`: live-joint and encoder capture, model
trajectory preparation, encoder profile preparation, final live-pose recheck,
and server processing time. Nested `v2_prepare` timings split trajectory
generation, route validation and velocity retiming. The UI shows total plan
processing time plus generation, route validation, retiming and encoder
preparation when available. The total also includes live reads and plan checks;
execution status retains the timings after consuming the plan.
Server timing excludes browser rendering and manual preview/review time.

Pinned v2 CUDA inference replays the original eager denoiser kernels through a
single cached CUDA graph for the normal four-candidate batch. DDPM scheduling,
random draws, precision, checkpoint, 32 denoising steps and collision tolerances
are unchanged. CPU and other batch shapes retain eager inference. The first
forward is compared exactly with eager output; a mismatch or a recognized
unsupported-capture error disables replay. Other CUDA faults still abort planning.
Set `MYCOBOT_CUDA_GRAPH=0` before starting the server to use eager inference for
diagnosis. Model loading and graph warmup remain lazy and require no robot IO.
This uses raw CUDA graph replay, not `torch.compile`: the compilation experiment
changed intermediate predictions and was not enabled.

The policy also caches the previous-timestep lookup within each denoising loop.
It reuses the known next scalar from the active schedule while leaving the
Diffusers DDPM step, coefficient arithmetic, random draws and timestep devices
unchanged. The cache is scoped to a single sampling call and restores the
original scheduler method even on failure. Unsupported scheduler versions,
custom schedules and unrecognized timestep objects use the original lookup.
Set `MYCOBOT_SCHEDULER_CACHE=0` before starting the server to disable this cache
for diagnosis; it is independent of `MYCOBOT_CUDA_GRAPH`. Direct Python callers
can use `DiffusionPolicy(..., scheduler_cache=False)`. The first implementation
is guarded for the installed Diffusers 0.40.0 scheduler; upgrading Diffusers
requires renewed exact-output qualification before expanding that guard.
See [scheduler comparison commands](../benchmarks/README.md) for offline checks.

The October 7 cache comparison held CUDA graph inference constant and pinned
benchmark workers to CPU 4. Ten ABBA blocks (20 warm measurements per setting)
supported approximately 5% faster offline planning on the recent-start and short
blocked cases; median times were 1624 → 1531 ms and 736 → 698 ms respectively.
All nine quick cases retained exact historical outputs and expected rejections.
These results describe instrumented offline planning under that test setup,
not browser/network or physical execution latency. Raw evidence and confidence
intervals are in [the timing report](../output/scheduler_lookup_optimization_20261007/timing-reviewed/report.html).

The subsequent collision joint-reset comparison kept CUDA graph inference and
the scheduler cache enabled in both settings. Ten ABBA blocks on CPU 4 supported
6.7% lower paired planning latency for the recent-start case and 7.0% for the long
blocked route. Warm medians were 1523 → 1427 ms and 5914 → 5503 ms respectively.
All nine quick cases and both timing cases retained exact historical outputs;
collision sampling and clearance thresholds are unchanged. These are offline,
instrumented planning measurements. See the
[collision timing report](../output/collision_validation_optimization_20261007/timing/report.html)
and [comparison commands](../benchmarks/README.md#comparing-collision-joint-resets).

Trajectory construction batches independent de Casteljau subdivisions while
retaining the original midpoint arithmetic, depth and derivative reductions.
Unsupported or extreme bound inputs use the original recursive calculation.
Final diffusion output frames use the existing URDF batch FK in blocks of up to
256 poses; positions, orientations and error calculations keep their original
definitions. `MYCOBOT_TRAJECTORY_BATCH=0` before process startup selects recursive
bounds and scalar frame FK for comparison. This setting changes neither curve
controls nor durations, motion limits, collision samples or model inference.

The October 7 trajectory comparison held all earlier optimizations on and pinned
workers to CPU 4. Ten ABBA blocks supported 5.5% lower paired planning latency
for the recent-start case and 9.3% for the long blocked route; warm medians were
1436 → 1357 ms and 5548 → 5037 ms. All nine quick cases and both timing cases
retained exact historical outputs. These results measure the combined bound
and frame-FK optimization in instrumented offline planning. See the
[trajectory timing report](../output/trajectory_construction_optimization_20261007/timing/report.html)
and [comparison commands](../benchmarks/README.md#comparing-trajectory-construction).

Collision path checks batch the original connecting-edge samples and identify
dense edges without scalar dispatch; frame metadata reuses its FK transform.
Before/after timing, exact trajectory/sample comparisons, held-out scenes and
encoder-packet comparisons are recorded in
`output/planner_optimization_20261007/benchmark.json`. The offline harness is
`tmp/benchmark_planner_20261007.py`; it does not open hardware transport. Timings
describe warm local planning, not physical execution or browser/network delay.

In **Fixed target on the arm**, select **v2 velocity profile · encoder + joint
speeds**, then use **Reference velocity shape**. **Original bell** preserves the
saved quintic reference. **Adjustable ramps** exposes acceleration and
deceleration time sliders (5–80% of each retained leg; their sum must be at most
100%). The UI defaults to Adjustable ramps with 5% acceleration and 10%
deceleration, leaving an 85% constant-speed plateau. The small preview
shows relative speed immediately; click **Plan from live joints** to generate
and review the actual signed J1–J6 profiles. Editing the shape invalidates the
previous plan and disables Execute until replanning.

The adjustable reference uses smooth cubic velocity ramps, with unit area and
zero velocity/acceleration at each endpoint. All six joints share the progress
law, retaining the checked v2 path, corner stops and endpoint. Timing checks
analytic peak velocity and acceleration against the selected per-joint allowance
and the conservative arm/flange envelope. The UI sampler and encoder executor
use the same progress law and its derivatives. Shape settings are applied after
v2 generation and never passed into the neural policy. This feature applies to
fixed-v2 velocity-profile mode; mouse tracking keeps its current C2 references.

The encoder target looks ahead by up to 0.2 seconds on the current checked
leg. When live lag plus that lookahead would exceed the 6° command allowance,
the adapter shortens the lookahead along that same leg, including encoder
rounding in the bound. It keeps the original reference clock and analytic
speed words. The 5° reference-tracking stop remains independent; the command
allowance retains 1° of headroom for lookahead and encoder rounding.

The clock advances with actual monotonic elapsed time, with no arrival waits
between updates and no added 2° stages. A common cadence stretch, shown as
`cadence_time_stretch`, gives every original leg at least four 0.2-second updates;
the displayed preview and derivative profile use that same timing. Only the
original checked corners are retained. Tracking error above 5°, command lead
above 6°, missed corners, stale telemetry, mode/health/collision failures or
heartbeat loss send Stop instead of silently slowing the reference. The
endpoint must stop within 0.75° and pass the existing FK acceptance checks.
Total excursion has no cap beyond joint limits and collision checks; speed
allowance is 1–80%, with the UI arm-speed default set to 70%. Target presets
still cap the selected speed at 5%. Planning sends no motion. Physical speed units and
tracking quality require an operator trial.

The Pi bridge must include 0x3E in `NO_REPLY_COMMANDS` in
`hardware/pi_transport.py`; otherwise each asynchronous packet blocks later
telemetry for the 0.5-second ACK timeout. Install the updated bridge files as
described in `hardware/README.md`. Actual send intervals appear in the UI.
Legacy Classical IK and manual controls retain their existing drivers.
The earlier staged and small-trial drivers remain explicit selections.

Fixed v2 paths also offer **Encoder targets + individual speeds · trial** in
Advanced controls → Fixed target on the arm. This selects the `set_encoders_drag`
packet instead of `send_angles`. Learned IK + RRT* and diffusion v2 mouse
tracking now use the shared encoder velocity-profile executor described below.
Trial plans allow at most 20 waypoints, 3° excursion from the captured start on
each joint, and a 1–5% reference-speed allowance. Planning reads live encoders,
checks each joint's positive or negative 2048-centred convention against joint angles, quantizes
targets at nominal 4096 steps/revolution and rechecks the quantized path against
the pinned joint/self/table scene. Execution checks the mapping again, uses the
existing confirmation/heartbeat/refresh-mode gates, and sends each packet once
without waiting for a motor ACK or falling back to another driver.

Each speed word is the absolute encoder displacement divided by its reviewed
waypoint interval, rounded to an integer nominal step/s. A stationary joint
keeps its encoder position with speed word 1; zero-speed words are avoided
because zero behavior is not established on this firmware. This is not a
signed velocity interface or a guarantee of physical speed/acceleration.
Actual feedback can extend the waypoint timing. Raw target/speed words are
recorded in execution samples; reference and estimated measured velocity
plots remain separate. `POST /api/ik/encoder-drag/preflight` with `{}` performs
only the idle encoder/angle consistency check and sends no motion.
Slow telemetry/health cycles (>2 s) abort the encoder trial before another send.
Directional jog remains blocked after the failed qualification incident.
Encoder direction is inferred only when one sign fits paired angle/encoder
readings. Verified nonzero readings are saved in
`hardware/encoder_direction_calibration.json`, scoped to the Pi address, port
and firmware. Near zero, a joint can reuse its previously verified direction;
the current telemetry must still pass the centred-encoder consistency check.
An unambiguous conflicting reading invalidates the stored robot calibration.
Without prior evidence, an ambiguous joint must remain stationary and a path
moving it is refused. The error names the affected joints. Preflight, planning
and execute-time mapping checks send no calibration motion.
Discard the saved calibration when replacing the arm or changing servo
zero/encoder calibration; the endpoint and firmware are its storage scope.
These model choices come from saved simulation evaluations, not measured
physical accuracy.

**Continuous encoder trajectory · trial** adds feedback-paced streaming for
fixed v2 plans. It samples each checked rest-to-rest quintic, sends six position
targets and six nominal step/s magnitudes together, and replaces interior
targets while the arm is moving in refresh mode 1. It retains all reviewed
corners and observes each stopped corner/endpoint within 0.75° for at least
one second, with no more than 0.1° drift during that observation. It does not
wait for a stop at interior samples.

This mode keeps the 3° excursion and 1–5% allowance, and adds a 1°/s analytical
reference cap, at least two seconds per leg, at most five sends per second,
200 packets/60 seconds of reference motion, and the original 20-source-waypoint
limit. Planning suppresses repeated quantized interior targets and rechecks
every quantized segment. Live command lead is at most 1°; lag pauses the
sequence and an unresponsive waypoint times out with Stop. Measured excursion
above 3.25°, stale telemetry/health (>2 seconds), a health/mode fault or loss
of the browser lease aborts. Stop/disconnect cancel the sequence and restore
the leased refresh mode. All feedback and sends remain serialized.

Send intervals use actual monotonic time. Slow cycles stretch execution;
no samples are skipped and no catch-up burst is sent. Late sends can lower
the reviewed speed word, never increase it. The preview is the ideal checked
quintic; encoder quantization and firmware interpolation can differ. The live
status shows tracking gap, reference phase, actual send intervals and raw
target/speed words. Velocity charts use actual joint telemetry timestamps.
Use a fresh live pose, select this mode, plan/preview, then execute through the
existing frame, scene and movement confirmations. Development tests use fake
hardware; continuous physical tracking/speed fidelity requires a user trial.

**Staged encoder trajectory · unlimited excursion** accepts the complete
reviewed route without a fixed total-excursion or source-waypoint cap. Joint
limits and the original paired encoder mapping still bound every target. It subdivides the checked straight legs into stages of at most 2°
and keeps every original corner. Each stage uses the same feedback-paced
samples, 1° live lead, 1–5% allowance and 1°/s analytical reference cap.
Staged routes have no fixed 200-packet/60-second reference cutoff. Stage boundaries stop and undergo the
one-second/0.1° drift observation before checking the original encoder mapping
again. Their measured FK must match the reviewed quantized stage endpoint
within 5 mm and 3° tool-axis error before the next stage is sent. There is no
mapping rebasing or automatic replanning during execution.

The staged mode stops if measured excursion exceeds 3.25° from the last
accepted stage. The existing health, refresh-mode, heartbeat, collision and
10-second waypoint timeout checks still apply. Its execution deadline grows
with reviewed reference duration, packet count and stopped stages; it does not
apply the fixed 180-second cutoff of the small-trial drivers. The route-specific
deadline appears as `execution_timeout_s` in plan/execution status. The staged
review window is its reference duration plus 60 seconds, allowing the full
preview before expiry; execution still verifies the live start and encoder
mapping again. Other drivers retain their 60-second review window. Live status
reports completed stages. Removing the distance cap does not establish speed
or tracking accuracy over the full workspace.
Transport latency is unchanged, so actual execution may take longer than the
reference trajectory.
A supervised J1 −10° trial completed all 68 packets and five stages. Its
endpoint error estimated from live joint readings and URDF FK was 2.08 mm /
0.69°; these are not external pose measurements. Actual execution took 49.62 s
versus the 17.94 s reference, with a median 0.600 s send interval. The recorded
trial is in `output/encoder_staged_latest_live_result.json`; it supports bounded
fixed-target use, while reference-time fidelity remains unqualified.

**Target preset from live pose** in the same fixed-target panel offers J4 ±2°
and ±1.5°, plus J3/J1 ±1.5° presets under Small trials. Larger staged trials
offers J3 ±6° and J1 ±10° presets and automatically selects the staged driver.
Choosing a small preset while in staged mode restores the small streaming
driver. Selecting a preset reads fresh idle joints and
fills the corresponding FK-derived XYZ/reference tilt target, clears EEF
offsets, caps the selected speed at 5%, and invalidates the browser's previous
plan. Coordinates are regenerated from the current pose on every selection;
the presets do not replay saved absolute targets. Selection sends no motion
and does not automatically plan or execute. The v2 planner still checks the
complete route, encoder mapping, joint limits and collision guards before it can
be reviewed and executed. Presets are disabled during motion, planning, mouse
tracking or disconnection. `POST /api/ik/small-target` accepts only a listed
`preset` identifier and refuses moving/disconnected arms or joint-limit overflow.

**Held-out case** under Advanced controls loads one of the 24 saved XYZ/tool-Z
benchmark targets. In hardware mode, the model stays at the live pose and only
the target is copied; use **Plan from live joints** to check a new route. The
saved case start is never sent to the arm. Selection invalidates the previous
plan and sends no motion. Staged encoder mode has no total-excursion cap; a
case still needs a converged, collision-checked path and compatible live
encoders. The optional small-trial drivers retain their 3° excursion and
reference/packet budgets.

To reproduce the original held-out start/target pair, select **Simulation only**,
choose a Held-out case and use the simulation planner/playback. A hardware run
from a different live start is a run to that saved target, not a reproduction
of the original benchmark trial. Diffusion's three saved block examples remain
in its simulation-only selector; the hardware encoder scene has no block.

Open **Advanced controls** for target poses, joint editing, solver comparisons,
workspace clouds and diagnostics. **Tracking settings & details** exposes the
alternate mouse planner and execution diagnostics. The primary screen does not
run comparisons on startup. The direct `/?view=lab` link opens the complete
workbench and retains its initial simulation comparison. No view connects,
arms or starts physical movement automatically.

**Plan checked diffusion** is a fixed-target **simulation-only** preview. Its
status line explains why it is disabled: hardware mode, active mouse tracking,
another planner running, a missing checkpoint, or an XYZ-only target. Select
Simulation only, stop mouse aim and choose the 5D pose task to use it.
For real-arm v2 planning, select Diffusion v2 in Prepare the arm and use
**Advanced controls → Plan from live joints**. The link beside the disabled
simulation button opens those controls; it does not execute a path.

Simulation mode compares four classical solvers for the five-constraint task.
Position-only mode compares all eight earlier methods on an initial example. Choose target XYZ,
a held-out test case and a controller. **Compare all eight** uses identical
starting joints, speed cap and step budget. Click a result/legend to view its
trajectory; **Play** and the time slider animate the URDF. Error plots and the
saved benchmark include failed methods. Joint edits reset the starting pose.
Playback reconstructs joint angles by integrating each joint's velocity from
its segment anchor, so rendering and seeking do not accumulate drift. Checked
quintic/C2 trajectories use their exact velocity profiles; ordinary timed
waypoints use constant velocity along each adjacent segment. The velocity chart
shows signed J1–J6 curves in rad/s, with a separate combined speed readout.
Profiles use trajectory time; the playback multiplier scales preview time.

In **Simulation + hardware**, the **Joint velocity profiles** panel shows
signed J1–J6 reference and measured velocities in rad/s. Measured curves use
differences between consecutive live joint readings and actual elapsed time;
they are estimates, not direct motor velocity measurements. Learned IK + RRT*
and diffusion v2 mouse tracking send encoder targets with individual joint speed
words from analytic C2 derivatives through the shared encoder driver. Mouse
phase is paced by live feedback: timing lag pauses progress and is diagnostic;
actual-to-command lead stays within 3° and collision checks run every cycle. Legacy Classical IK
mouse tracking uses its bounded velocity servo.
Fixed-target plans show the planned analytic joint velocity profile. The default
encoder profile driver uses the same timed q(t)/dq(t)/dt reference during
execution, without adding stopped stages. Earlier waypoint drivers remain
telemetry-paced. Measured execution velocities use actual telemetry intervals. The hardware 3D view follows live joints.

Click **Plan self-checked PRM** to query the reusable roadmap with conservative
URDF self-collision checking. It uses the same target, local offset, start pose,
tolerances and solver budget as the lab. XYZ uses seed NN plus numerical
refinement/fallback; 5D defaults to its separate twelve-input Seed NN with four
bounded proposals, refined with the selected classical controller and bounded
DLS fallback. A numerical-only proposal option is also available. Missing or
incompatible 5D checkpoints are reported and numerical fallback still works.
The active 5D checkpoint is now the collision-aware fine-tune. The proposal
selector and mouse-aim engine identify it as collision-trained, and the status
readout shows its 1 mm self-collision training clearance. Training improves raw
endpoint proposals; endpoint and full-path collision checks still run before
playback. The previous checkpoint is saved under
`simulation/artifacts/seed5_v1/backups/pre-collision-webui-20261004-574a9107421b/`;
`simulation/artifacts/seed5_v1/activation.json` records the active model and hashes.
Enter tool Z as a nonzero base-frame vector (normalized automatically), choose
an axis/current-tool preset, or retain reference tilts. Goal acceptance requires
both position (mm) and direction (degrees) tolerances; local tool-Z spin is free.
Intermediate PRM motion may change pointing direction. See [SEED5.md](../simulation/SEED5.md)
for training/checkpoint provenance. A direct valid connection is
preferred; via points are used when required. Play or scrub the resulting preview
and inspect target/achieved tool directions, both errors, accepted-goal sources,
separate IK/planning outcomes, the joint/FK waypoint table and candidate/fallback reasons. Failed
queries report invalid starts, IK failure or path failure and do not enable a
preview path. Input edits discard the prior result.

Install `.venv/bin/python -m pip install -r simulation/requirements-collision.txt`
before using this button. The saved roadmap uses 1 mm clearance and 0.04 rad
per-joint segment sampling. The first query may take a minute or more while the
bridge revalidates the cache; subsequent queries reuse the graph. See
[self-collision assumptions](../simulation/SELF_COLLISION.md). Restart an older
bridge after updating and refresh the browser. PRM does not prepare a hardware
execution plan.

Browser regression checks use an isolated simulation backend and fake hardware
state; they do not contact the robot:

```bash
cd webui
npx playwright test tests/prm.spec.js tests/bounded-dls.spec.js
```

### Diffusion trajectory preview

The **Diffusion trajectory planner** panel runs the trained six-joint DDPM
alongside the existing controllers. Select a held-out example, press **Plan
checked diffusion**, then **Play** or scrub the result. The orange block and
table match the simulation collision scene. Position/size, sampling seed,
candidate count, denoising steps and decision budget are editable; custom 5D
targets first use scene-valid IK seeded from the actual current viewer joints.

Choose **Obstacle → No block** to remove the box for previews and hybrid mouse
tracking. Joint limits, self-collision and table clearance remain checked;
**Show diffusion scene** controls visibility only. Block fields are disabled in
this mode, and changing modes discards previews and stops mouse tracking.
The API requires explicit `obstacle_mm: null` for no block; omitting the field
is rejected. The model input then has zero block geometry and a zero presence mask.

The active `diffusion_motion_v2` checkpoint uses the 100,000-episode expert set
plus 20,000 split-preserving augmentations, including absent blocks, moving
goals and retargets with nonzero starting motion. Its inputs include joint
velocity/acceleration and Cartesian target velocity. All six full WebUI preset
checks (three with the block, three without) reached their goals. Failed
queries still return no playable full path. Moving training goals follow expert
routes rather than unrestricted mouse paths.

Each full 16-step proposal is checked for joint limits, self, table and any selected block
collision before speed projection/rechecking. The motion model then fits a
checked C2 curve through the full horizon and advances up to 0.4 seconds before
replanning, carrying its velocity and acceleration. Rejected replacements
retain a previously checked braking tail. The
completed path receives checked shortcuts, quintic timing and a final collision
check before playback. Failure holds the
preview pose and returns no playable partial trajectory. Displayed timing
uses 0.02-second references, with a 1 rad/s vector speed limit and 3 rad/s²
acceleration limit. Timing starts and ends at rest and stops at retained corners.
Dense route validation batches up to 256 poses for URDF forward kinematics,
collision bounding-box transforms and link-pair screening. It retains the
existing curve samples, interpolation grid, convex FCL checks, table/block
clearance and exact-pose cache. Native FCL queries run only for nearby pairs;
numerically ambiguous clearance boundaries use the original scalar predicate.
Custom validity callbacks retain scalar dispatch. This changes validation
processing, not the v2 trajectory or the selected hardware velocity shape.
Bullet scene validation submits the six revolute joint positions together,
with explicit zero velocities, using `resetJointStatesMultiDof`. The same
table/block clearance queries follow each pose, and the displayed pose is
restored after validation. `MYCOBOT_BULLET_BATCH_RESET=0` selects the original
six-call update for comparison; installations without the batch API use that
fallback automatically. Runtime Bullet errors still propagate.
Numerical/PRM actions hide the
diffusion scene. Input changes discard old and pending previews.

Playback evaluates the exact checked quintic at each render time instead of
jumping between stored frames. A single blue marker shows the final requested
target; **Show final target marker** toggles it. Intermediate trajectory points,
path trails and moving reference markers are not rendered.
The joint velocity plot and readout use trajectory time. Pause/resume retains
the continuous position, and delayed renders visit retained corners at rest.
Hybrid mouse chunks show the final requested target, even when the accepted
chunk ends before that target. The marker stays fixed during playback and changes
when a replacement path takes over.

This is available only with hardware disabled/disconnected. It does not
create a robot execution plan. The full checkpoint hash and scene are pinned
in `simulation/artifacts/diffusion_motion_v2/webui.json`, selected by the atomic
`simulation/artifacts/diffusion_active.json` pointer; incompatible installations
disable this planner while leaving the other controllers available. Install
dependencies with `.venv/bin/python -m pip install -r
simulation/diffusion/requirements.txt`, then rebuild/restart with
`./start_webui.sh`. See [the experiment and validation](../simulation/diffusion/README.md).

### ROS 2 detected-object tracking

[`object_tracking_ros2`](../object_tracking_ros2/README.md) replaces mouse input
with the accepted 3D center published by `objectScanner`. The ROS follower uses
the same feedback-paced hardware driver, a **350 mm / 30°** acceptance region,
and timestamped camera-to-EEF-to-base transforms. Detection loss holds the arm
in place while keeping the session armed; fresh accepted detections resume with
a new checked plan from measured joints. Explicit Stop or hardware faults
disarm the session. It starts disarmed and holds
one exclusive controller session. The existing WebUI remains available for
telemetry and Stop. Follow the linked launch, assumed-mount and arming instructions.

### Shared velocity profile for mouse tracking

The **Reference velocity shape** editor in tracking settings controls both mouse
tracking and fixed-target Velocity profile mode. The default adjustable ramps
use **5% rise / 10% fall**; the original bell profile remains selectable. Learned
IK + RRT* and diffusion v2 hardware tracking pin the chosen shape when a session
starts. Changing the shape requires stopping and starting tracking again.
Simulation tracking also applies this editor, including the legacy IK planner.

The main **Joint velocity profiles** panel renders the latest accepted mouse
path with the same signed J1–J6 chart used for fixed-target plans, in rad/s over
trajectory time. The current curve stays visible during replanning and changes
when the replacement takes over. Hardware tracking also shows the separate
commanded and measured curves over actual elapsed time. Simulation displays
only the planned curve. Stopping or invalidating tracking clears that curve.

Resting trajectory legs evaluate the same `simulation.reference_shape` progress
law as fixed-target moves. Hardware cursor replacements reanchor to fresh live
joints and speed-capped measured velocity, then finish at rest. Their checked
references report `collision_checked_velocity_profile_c2` and the selected
`reference_shape`; derivative bounds and self/table/path replay guard the whole
curve. Both hardware engines use `webui.encoder_profile.run` and nominal 200 ms
encoder sends. Mouse reference phase follows live feedback, with a 3° command
lead bound; lag stretches execution instead of triggering a 5° timing stop.
Fixed-target velocity mode retains its strict clock and tracking-error guard.

The endpoint still requires 0.75° joint settling and disk/cursor acceptance.
A settled mouse endpoint outside acceptance requests a live-start correction.
Self/table collision checks, joint limits, health, lease and Stop remain active.
Short legs retain at least four nominal send periods. Encoder calibration is
captured from idle telemetry, checked before execution and reused across cursor
updates. Replacements hold the last checked command during planning, activate
only after acknowledgement and a bounded live connection, and preserve one
execution owner. Expired or drifted anchors request replanning.

Hardware requests include `velocity_shape` on `/api/ik/mouse-aim/start`; planning
must retain that session setting. The browser requires
`mouse_velocity_profile_protocol: 1`, `mouse_encoder_profile_protocol: 1`,
`mouse_feedback_tracking_protocol: 1` and
`mouse_executor: encoder_feedback_tracking`. Plans identify
`execution_mode: encoder_feedback_tracking` and `joint_command: encoder_drag_profile`.
The planned chart uses reference time; measured curves and gaze errors use actual
elapsed telemetry time. Legacy Classical IK retains its existing driver.

### Learned IK + RRT* mouse tracking

Both simulation and hardware tracking selectors default to `learned_rrt` when
its pinned learned IK checkpoint is ready. `/api/sim/mouse-aim` dispatches
`tracking_engine: "learned_rrt"` to the shared classical tracker. Hardware starts
through `/api/ik/mouse-aim/start` with the same engine, table confirmation and
speed settings. Starting joints and derivatives belong to the server;
client-supplied joints cannot become hardware references.

The tracker finds a scene-valid endpoint with the evaluated current/learned/random
virtual-start solver (32 attempts and robust numerical fallback). It refines
cursor gap/direction when necessary, checks a direct route and uses bounded RRT*
for detours. Simulation curves preserve position, velocity and acceleration at future
retargeting joins. Hardware uses acknowledged live-feedback anchors. Both end
at rest and retain the prior checked path if a replacement fails. Simulation also supports bounded resting chunks and an
optional table-mounted block; hardware mouse tracking uses no external block.
Joint limits, self and table collision checks, telemetry, heartbeat and Stop
remain active. The fixed-target benchmark does not measure tracking accuracy.

### Hybrid diffusion mouse tracking

**Continuous motion** is enabled by default for both simulation mouse planners.
Local IK now supplies the complete checked path to the cursor goal. The viewer
continues following it while a replacement is computed from a future reference,
including that reference's velocity and acceleration. A ready replacement joins
at that exact point with continuous position, velocity and acceleration; cursor
updates coalesce while one request and one future replacement are pending.
Checked Bezier curves cap vector speed at 1 rad/s and acceleration at 3 rad/s²,
and are collision sampled at 100 Hz before returning 0.02-second references.
Their joint and derivative control hulls bound the entire continuous curve.

The path still finishes at rest at its goal and at retained collision-route
corners. If a replacement is late or rejected, the current checked path continues
through its braking tail; a late interior join is discarded rather than jumped
to. Stop, Escape, scene edits and leaving the viewport cancel immediately.
Hardware mouse control keeps its existing behavior and hides this checkbox.
Turn off **Continuous motion** to use the earlier stop-before-replanning behavior.

Select **Local IK → diffusion · self / table / block** above the viewport and
start mouse aim. Local IK uses the captured current joints first. Its full path
and diffusion proposals use the displayed diffusion table/block collision scene.
With Continuous motion off, the next two diffusion references or up to 0.2 seconds of local IK are smoothed
and retimed into a checked chunk lasting at most 1.3 seconds. Normal retargeting
waits until it finishes at rest, then replans the latest cursor from the exact
current joints. Pending responses for changed targets
are discarded, and scene/settings changes stop tracking. The diffusion decision
budget limits repeated queries for one cursor position.

The default Seed NN / PRM tracker and hardware controller remain available.
Hybrid mode uses the motion-trained checkpoint. Continuous tracking sends the
predicted starting velocity/acceleration and a bounded target-velocity estimate.
With Continuous motion off, the neural fallback retimes a checked geometric
prefix of the full horizon into a resting chunk. Some targets still fail or
stall. Planned references cap speed at
1 rad/s and acceleration at 3 rad/s²; immediate Stop can interrupt a chunk.
See [mouse tracking](../simulation/MOUSE_AIM.md) and
the [paired model comparison](../output/diffusion/motion_v2_continuous_comparison.json)
for the latest diffusion evaluation. Both checkpoints used the same C2 controller:
fixed-goal success was 29/64 → 63/64, no-block success 0/32 → 32/32, and
moving/retarget success 11/24 → 23/24. Moving-target RMSE over the same 48 seconds
was 42.0 → 31.7 mm. These pure-policy suites bypass local IK and exclude planner
compute delay; they do not measure browser cursor latency. CPU denoising remains
about 64 ms for four candidates and 32 steps.

### EEF local target offset

The target panel accepts local EEF X/Y/Z offsets in millimeters. The resolved
flange target is `base XYZ + R_base_from_starting_EEF × local offset`. Rotation
comes from all six starting joints, including flange-Z spin, in both task modes.
Connected simulation uses live joints; hardware planning recomputes the rotation
from fresh live starting joints on the server. Target tilt does not rotate this
offset. The resolved position, target axes and orange marker show that result.
Playback preserves the starting offset frame. Editing an offset invalidates a
prepared hardware plan. Loading a test case or using the live flange pose resets
offsets to zero. This shifts the requested flange position; it does not change
the robot model's tool center point.

## Phase 2 hardware IK

For **diffusion v2 hardware**, run
`./start_v2_hardware.sh` and follow [v2 hardware instructions](HARDWARE_DIFFUSION.md).
The tracking selector defaults to learned IK + classical RRT*. Use speed 5
for initial mouse trials; the shared speed field can be adjusted up to 80. Planning reads fresh stopped
joints; execution requires path, frame and table verification. Classical IK and
hardware mouse aim remain available. v2 mouse tracking now shares the continuous
simulation planner and checked C2 curves, with the shared fixed-target encoder
velocity executor for hardware execution.

Hardware control is opt-in with `./start_webui.sh --hardware`.
See [phase 2 instructions](PHASE2.md) for planning from live joints, reviewing
the predicted path and executing bounded joint waypoints with telemetry.
This is separate from the default phase 1 simulation workflow. In hardware mode,
run the [updated Pi bridge](../hardware/README.md) on port 9000.

### Mouse control of the arm

In hardware mode, the viewport shows **LIVE ROBOT CONTROL**. Connect, enable
movement, verify the flange frame and table using the checkboxes, retain
**Learned IK + classical RRT*** and set **Arm speed 5**, then click
**Start robot mouse aim**. Moving over the scene requests a 50 mm cursor gap
with the flange pointing toward it. The server uses current joints, learned virtual IK starts and bounded numerical
refinement to find a cursor endpoint, then checked direct/RRT* planning. It
independently checks the continuous no-block curves,
and delivers them through the fixed-target encoder/individual-speed executor.
Replacements match
position, velocity and acceleration at server-owned future joins. Partial
chunks continue planning even for a stationary cursor. Firmware refresh mode 1
is verified. The reference clock keeps advancing; missed corners or excessive
tracking lag stop execution. The final goal must settle with stopped telemetry.
Diffusion v2 remains selectable with local IK and diffusion fallback. The
legacy Classical IK option retains the IK/Seed5/PRM velocity controller. The model displays
actual telemetry and the UI reports verified mode and reference limits.
Leaving the scene or Escape stops tracking and restores the original refresh
mode. See [hardware mouse aim](HARDWARE_MOUSE_AIM.md)
for operation, API, cancellation and validation details.

### Connect and move

1. Enter the Pi's IP address (default `192.168.77.225`) and TCP port (`9000`).
2. Click **Connect**. This reads firmware, joint angles and motion status; it
   does not power up servos or send movement commands.
3. The model initially follows live readings. Uncheck **Follow robot** to edit
   a preview with joint sliders, angle fields, Zero, or Rest. **Use live pose**
   copies the actual angles into the preview.
4. Clear the arm's path and check **Enable movement — path is clear**.
5. Click **Move robot to preview pose** to send all six target angles at the
   selected speed (default 30; range 1–80).
6. Alternatively, select a joint and click **− Jog** or **+ Jog**. This moves
   by direction and speed for 0.1–2 seconds (default 0.5). The Python bridge
   sends stop when the timer expires, even without further browser requests.
7. **Stop robot** or Escape sends a stop command. Disconnect and normal bridge
   shutdown stop any movement started through this bridge before closing TCP.

Live angles are displayed separately from preview angles and refreshed about
once per second. Follow mode displays actual angles even when they are beyond
URDF display limits. Motion requests use the SDK's robot joint limits and check
power, servo enable, pause and error states. The server rejects overlapping
movements. Old Atom firmware may not acknowledge movement packets, so the
bridge sends once and checks telemetry instead. A pose move exceeding 90
seconds sends a stop command and reports a timeout. Read failures attempt stop
and require an explicit reconnection; movements are not replayed.

The Stop button is a network software stop, not a hardware emergency stop.
It depends on the PC, network and Pi server; delivery cannot be guaranteed
when those fail. The viewer does not check collisions or confirm a safe path.
Rest is a custom drawing-demo pose, not a manufacturer parking pose.

## Viewer controls

- In simulation mode, click **Start mouse aim** above the 3D scene, then move the
  mouse over the scene. The cursor is projected onto a camera-facing plane through
  a point 50 mm along the starting flange Z axis. The plane stays anchored at that
  depth and faces the camera when the view changes. The flange Z axis aims at the
  cursor while local 5D IK first seeks a 50 mm gap from the current joints.
  The selected 5D controller is honored; XYZ experiment mode uses 5D DLS for
  mouse aim. The backend validates the complete local path, simplifies it only
  through checked shortcuts, and uses trained Seed NN / numerical / PRM recovery
  when local motion cannot be accepted. It validates the speed-limited
  trajectory before playback. A blue cursor marker, purple desired flange position and dashed distance
  line show the target; readouts show the actual gap and pointing error. Targets
  that cannot converge within the current limits retain their visible error.
  Tracking pauses outside the scene, on window blur, or while dragging to orbit.
  **Stop mouse aim**, Escape, pose/settings edits, experiments and playback stop
  tracking and discard late results. The browser holds the preview while planning
  and combines cursor updates into one latest target. Local solver iterates become
  motion only after complete path validation and simplification; fallback
  refinement iterates remain computations. Direction is constrained at each goal, with local tool-Z spin free;
  intermediate motions may change direction. Only the simulation endpoint is
  called, and both the server and UI require hardware disabled and disconnected.
  Readouts identify local IK or fallback goal sources, checkpoint status when used, IK/route outcome,
  goal errors and actual cursor gap/angle. Position and direction tolerances,
  speed, damping and step budget use the experiment settings above, without hidden
  tolerance tightening. Actual cursor gap/angle must pass before a goal is accepted.
  **Compare this target** transfers the exact captured start and 5D target into
  those controls. Failure feedback separates collision rejection, cursor geometry
  and exhausted numerical search. A missing
  compatible 5D checkpoint uses numerical fallback, never an XYZ checkpoint.
  See [mouse-aim workflow and limits](../simulation/MOUSE_AIM.md).
- Drag to orbit; Shift + drag or right drag to pan; scroll to zoom.
- Frame, Front, Side and Top change the camera.
- Toggle grid, axes or wireframe. Hover a joint row for its coordinate axes.
- **Camera FOV** draws an amber rectangular pyramid from the wrist's configured
  optical mount, using the current horizontal/vertical FOV and near plane. The
  illustration extends to 500 mm; that drawn length is not a sensor range limit.
  A dashed ray and angle readout show the active object's image-bound projection.
  Red means outside the image bounds. Occlusion is checked by the planner.
- Copy exports preview angles as a JSON array in J1–J6 order.
- Download URDF saves the original file. Its meshes are also needed elsewhere;
  the complete local package is in `public/models/mycobot_description`.

## Model provenance

Assets are bundled from
https://github.com/elephantrobotics/mycobot_ros/tree/noetic/mycobot_description/urdf/mycobot_280_pi
with revision and license in `public/models/SOURCE.txt` and `public/models/LICENSE`.
No CDN or ROS installation is needed to display them. The upstream URDF calls
its robot `firefighter`; the UI labels it myCobot 280 Pi. It has eight links and
six movable joints displayed in chain order.

The viewer normalizes numeric tuple whitespace and converts COLLADA polygon
faces into polylists in memory. A stale wrist material reference is rebound to
its single defined material. This restores the wrist and Pi housing while
preserving the downloaded URDF, geometry and dimensions. URDF and firmware
joint limits may differ.

## Verify

```bash
.venv/bin/python -m unittest discover -s webui/tests_python -v
cd webui
npx playwright test
npm run build
```

Python tests use a fake robot to verify validation, timed stops, failed sends,
connection loss, disconnect, command completion and HTTP request protection.
Browser tests use installed Google Chrome. Hardware tests mock robot API calls;
simulation tests use the localhost simulation API with hardware disabled. No
movement commands are sent to hardware.

The dry-run `/api/sim/prm` API adds reusable joint-space via-point planning using
an offline roadmap. It returns the existing simulation frame format and reads
live joints when connected, without preparing an executable hardware plan. See
[PRM instructions and benchmarks](../simulation/PRM.md).
PRM queries now include conservative self-collision checking by default; the
existing solver comparison and hardware execution paths remain separate. See
[self-collision setup and assumptions](../simulation/SELF_COLLISION.md).

## Current fixed-v2 benchmark

The restored continuous C2 velocity-profile method is saved in
[`benchmarks/current.json`](../benchmarks/current.json). Its referenced snapshot
contains profile comparisons, timing measurements, test results and physical
trial outcomes. The timestamped backup includes the pinned model and runtime
source. Preserve this benchmark when comparing later controller changes.
