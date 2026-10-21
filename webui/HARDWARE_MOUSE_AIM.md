# Hardware mouse aim

## Cone and disk acceptance

Learned IK + classical RRT* and diffusion v2 mouse tracking use the cone/disk
strategy described in [simulation mouse aim](../simulation/MOUSE_AIM.md).
Defaults are **100 mm axial distance** and **20° half-angle** (approximately
**36.4 mm disk radius**). Adjust them in **Acceptable tracking area** before
starting; changing either stops the session.

The server captures the cone axis from live EEF telemetry toward each new
cursor target and holds it fixed for that target. It reconstructs the region
from validated settings and returns its authoritative geometry to the browser.
The planner chooses an EEF endpoint inside the disk, with local Z pointing at
the cursor from that endpoint. Region settings are pinned for the session.
Endpoint completion checks plane distance, disk radius and cursor-facing Z
using the existing hardware **5 mm / 3°** tolerances; joint settling and all
collision checks still apply. Off-center endpoints are not required to have
exactly 100 mm Euclidean distance to the cursor.

The browser requires `mouse_tracking_region_protocol: 1` for this strategy.
Both engines share the encoder packet driver and velocity shape with
fixed-target velocity mode. Mouse tracking advances its reference from live
feedback; fixed targets retain their strict trajectory clock. Classical IK retains its previous
point-target controller. Validation uses simulated hardware; it does not measure
physical tracking accuracy.

The viewport uses simulation mouse aim in the default server mode and physical
robot mouse control when the bridge runs with `--hardware`.

## ROS 2 detected-object input

The separate [`object_tracking_ros2`](../object_tracking_ros2/README.md) adapter
uses accepted `/object_pose/filtered` box-center poses from `objectScanner`
instead of mouse coordinates. It transforms camera optical points at their
source timestamps using measured EEF feedback, then uses this same checked
hardware planner/driver with a **350 mm / 30°** region. Its camera mount defaults
to the requested assumption: zero optical-to-EEF offset with optical Z aligned
to tool Z. ROS and WebUI tracking retain exclusive session ownership. Run
`./start_object_tracking.sh --hardware` from the project root, then arm through
`/object_follow/arm` with a fresh detected target. Starting the adapter is disarmed.

## Use

1. Run the updated [Pi bridge](../hardware/README.md), then
   `./start_v2_hardware.sh` and open http://127.0.0.1:8000. Connect to the Pi's
   current address (default `192.168.77.225:9000`). Startup sends no connection
   or motion commands. Reload the browser after upgrading the PC server.
2. Use the existing frame-verification and **Enable movement — path is clear**
   checkboxes in **Prepare the arm**. Keep the default **Learned IK + classical RRT*** or select **Diffusion v2**, verify the displayed
   table at **−35 mm in `g_base`**, and check the scene confirmation. The first
   scene has no block; tools, cables and other external obstacles are unmodeled.
   Start at **Arm speed 5** (mouse tracking allows 1–80%). Experiments and fixed
   targets are under **Advanced controls**; tracking diagnostics are under
   **Tracking settings & details**.
3. Click **Start robot mouse aim**. Starting reads live joints, checks robot
   health, captures the verified encoder mapping and records the command refresh mode; it sends no motor command or
   mode change. Tracking is refused if the mode cannot be read reliably.
4. Move over the scene. The cursor lies on a camera-facing plane anchored
   50 mm along the starting flange Z axis. The arm aims flange Z toward the
   cursor and selects an endpoint in the 100 mm / 20° disk. Classical IK
   retains the legacy 50 mm point target. Coordinates are `g_base`, the controlled point is
   `joint6_flange` with zero tool offset, and local-Z spin is free.
5. Leave the scene or press Escape to stop. Orbiting, focus loss, hiding the
   page, input changes, disarming, Stop and Disconnect also cancel tracking.
   Start again explicitly to resume.

## v2 continuous tracking

The v2 selection uses `simulation.diffusion.tracking.plan` and the same checked
quintic C2 curves as continuous simulation mouse tracking: local pose IK first,
then the pinned `diffusion_motion_v2` policy if needed. The checkpoint remains
`6582cbf4e1d2575007cfead8f99ed5f15a7b7021083d0eb200c61b224401bddc`, schema 2.
No training assets change. Cartesian v3/v4 are not accepted by this adapter.

The objective is to keep tool Z aimed at the object without self-collision.
The scene is static; accurate arrival at each trajectory timestamp is not a
tracking requirement. The 100 mm / 20° disk remains the desired endpoint region.
Planners report `tracking_objective: gaze_first` and per-frame cursor-facing
errors; diffusion fallback ranks gaze ahead of disk displacement and joint travel.
Orientation is an endpoint constraint, so a large cursor jump or collision detour
can temporarily increase gaze error.

The server owns every starting position, velocity and acceleration. Initial
planning starts at fresh stopped telemetry. During motion it holds the last
checked command while planning from fresh live joints, retaining measured,
speed-capped starting velocity to avoid restarting the ramp on every update.
A replacement keeps the execution owner and needs a separate browser
acknowledgement. It activates on a telemetry cycle only if its live anchor is
within the 3° bounded connection. Rejected or expired proposals retain the
previous checked path and request a fresh plan. Playable partial chunks request
continuation even when the cursor is stationary.

Before accepting a curve, the hardware adapter independently checks all joint
control bounds, derivative bounds, internal C2 joins, resting tails and self/table
collisions. Curves are sampled at 100 Hz and connecting edges at 0.01 rad with
the pinned collision model. Execution repeats live-pose and live/reference-to-
command collision checks. Neither the browser nor API callers submit joint paths.

Learned IK + RRT* and diffusion v2 tracking call the same
`webui.encoder_profile.run` executor as fixed-target Velocity profile mode.
Both send `set_encoders_drag` once per nominal **200 ms** period, using individual
joint speed words from instantaneous analytic velocity. Targets use the same
bounded 200 ms lookahead and verified encoder direction/offset mapping. The
mapping is captured from stopped telemetry and verified again before execution.
Unknown moving-joint encoder directions are rejected; no angle-command fallback
is used. Short legs are regenerated with at least four nominal send periods and
independently checked while preserving the starting q/v/a boundary.

The existing tracking reference vector speed remains the smaller of the selected
simulation speed, 1 rad/s and `20 × Arm speed / 30` degrees/s. Acceleration remains
bounded by 3 rad/s². The hardware command driver changes; these conservative
tracking planner caps remain. The selected profile defaults to 5% rise / 10% fall.

Mouse reference phase advances only while actual joints remain close enough
for a checked command, with at most **3° actual-to-command lead** including
encoder rounding. Motor lag pauses phase instead of causing the fixed-target
5° reference-error stop. Original route corners wait for live joint arrival
before proceeding. The endpoint still settles within **0.75°** for **0.5 s**
and passes the existing **5 mm / 3°** disk/cursor acceptance. A 180-second budget
applies to each accepted cursor curve. Partial prefixes request continuation rather than claiming arrival. A settled
mouse curve that misses disk/cursor acceptance also requests a correction from
fresh live joints; the UI retains the measured error and does not claim arrival.

Nominal curve duration and analytic velocity remain visible in the planned
profile. Actual execution can take longer; the UI displays `reference_lag_s`
and live cursor-facing error. Actual-to-command collision checks run every
cycle, including the bounded connection to an acknowledged live anchor.

Telemetry, health and refresh mode are checked each send cycle. A cycle over
two seconds, stale five-second browser lease, invalid live pose/path, excessive
command lead or unreadable telemetry stops execution. Encoder speeds remain nominal
step/s magnitudes; measured velocities use live readings, and FK/cursor errors
are URDF estimates. The model continues displaying live telemetry.

The browser requires `mouse_encoder_profile_protocol: 1`,
`mouse_feedback_tracking_protocol: 1` and
`mouse_executor: encoder_feedback_tracking` before owning a continuous tracking session,
plus the existing engine/profile protocols. Plans must identify
`execution_mode: encoder_feedback_tracking` and
`joint_command: encoder_drag_profile`.
An older bridge cannot silently execute the previous angle-target driver.
SDK/reference-speed or profile edits stop the session; start again to change it.

## Classical IK tracking

Select **Classical IK** to retain the preceding feedback velocity controller.

The server reads fresh live joints and first tries the selected local 5D IK
controller from that pose. The entire local path, including intermediate
configurations and segments, must pass self-collision and joint-limit checks.
Position, directed tool-Z and actual cursor geometry must pass goal acceptance.
Freshly checked shortcuts simplify solver ticks into route vertices, so every
small IK update does not become a mandatory hardware stop. Preview frames are
retimed to the existing speed bound and validated again.

If the local solve stalls, exhausts its budget, misses acceptance or has an
invalid path, the existing trained Seed5, numerical/IK coverage recovery,
direct/PRM and optional guided-repair pipeline runs. The roadmap loads only
when fallback is needed. Both paths retain 1 mm conservative self-collision
clearance, operating joint limits and 0.04 rad per-joint edge sampling. Simulation
classical mouse tracking uses this same local-first policy. The UI reports the chosen
route, and API results include `tracking_source` and `local_ik` diagnostics.
Model assets are unchanged.

The browser supplies a cursor and settings, never starting joints or motor
trajectories. It validates returned predicted frames and executes only a
one-use server plan ID. Predicted frames remain available for validation. The
mouse worker instead integrates a persistent, feedback-limited joint velocity
into short position references using fresh joint telemetry. Reference speed is
limited to 20 degrees/s per joint at SDK speed 30 (6.7 degrees/s at speed 10),
and reference acceleration to 60 degrees/s². SDK speed remains a separate
firmware motor setting. These limits bound the requested reference profile;
they do not guarantee physical motor speed or acceleration.

The reference normally leads actual joints by at most 3 degrees, with a hard
4-degree live-to-command bound. Both the previous-reference-to-command and
live-to-command segments pass joint-limit/self-collision checks. Original
future route vertices are retained; turns and the final goal require stopped
telemetry within 1 degree of the vertex. A replacement route may skip its
already-passed prefix only through a checked join within 1 degree of the
current reference and 4 degrees of actual joints. Health, drift, heartbeat,
Stop and timeout checks remain active. The model displays actual telemetry.

The mouse worker waits 50 ms between telemetry iterations, plus the actual SDK
round-trip and checking time. Integration uses the measured elapsed time,
capped at 150 ms; a cycle longer than two seconds stops execution. The controller
poller avoids duplicate SDK reads while the acknowledged worker is reading
telemetry. Execution is acknowledged as soon as its response reaches the
browser, rather than waiting for the next one-second lease heartbeat.

The PC reply reader reconstructs complete TCP packets and ignores unrelated
acknowledgements without discarding following telemetry. It allows an 800 ms
reply window for the legacy Pi server. The updated
[Pi bridge](../hardware/README.md) splits incoming TCP packets and forwards
this arm's unacknowledged motor/Stop/mode commands without waiting for a reply.
An unchanged reference is not sent as a motor target. Cycle and telemetry read
times are retained in execution diagnostics.

After that first acknowledgement, the worker sets and verifies refresh mode
**1** before any motor target. The manufacturer defines mode 0 as queued
execution and mode 1 as replacement by the newest command; see the
[myCobot 280 Pi Python API](https://docs.elephantrobotics.com/docs/mycobot_280_pi_en/3-FunctionsAndApplications/6.developmentGuide/python/2_API.html).
The mode is checked every 500 ms during execution. A failed query or unexpected
change stops tracking. Stop, session expiry and Disconnect attempt to restore
the recorded original mode after stopping motion; failed restoration closes
the connection. The UI shows the verified mode and current reference limits.

At most one worker executes. New pointer targets can be planned from live
joints while the owned mouse route moves. A valid replacement retains the
worker's execution ID, reference and velocity, and changes route only after a
separate acknowledgement of its one-use plan ID. Unacknowledged replacements
expire after two seconds. A rejected or stale candidate leaves the last checked
route running, which the UI states explicitly. Periodic execution heartbeats
cannot authorize a replacement.

Pointer updates coalesce to the latest target, with a 1 mm update threshold
and at least 150 ms between accepted plans. Candidates more than 10 mm behind
the latest cursor are discarded before execution. A mouse session reserves
the controller against manual move/jog/planning. Stop invalidates its prepared
and pending plans and in-flight planning work. A late response cannot authorize
motion, and a late stop from an old session cannot interrupt a new session.
The browser requires tracking protocol 2. A v2 selection additionally requires
v2 tracking protocol 1 and the matching engine; an older PC backend is refused
until restarted rather than silently using Classical IK.

Planning defaults remain 1 mm / 1 degree. Physical completion uses the
existing 5 mm / 3 degree URDF-FK estimates, a 1-degree classical final joint arrival bound,
and the actual live cursor gap and direction. Hardware requests cannot exceed those planning
tolerance bounds. These estimates use joint telemetry and the URDF, not an
external Cartesian measurement.

## API and leases

Requests require the existing local-session token and JSON content type.
Hardware-disabled servers reject all hardware mouse endpoints.

| Endpoint | Accepted data | Effect |
|---|---|---|
| `/api/ik/mouse-aim/start` | `armed`, `frame_verified`; v2 also `tracking_engine=diffusion_v2`, `scene_verified`, SDK `speed`, optional `speed_rad_s` | Checks readiness and records the original mode; no motion or mode change. |
| `/api/ik/mouse-aim/plan` | `session_id`, `cursor_mm`, solver and guided-repair settings | Read-only live-start planning, including during owned mouse motion; returns predicted result and server plan ID. |
| `/api/ik/execute` | Existing plan ID, speed, movement/frame flags; v2 also `scene_verified` | Consumes a plan; starts one worker or prepares an acknowledged replacement on the same execution ID. |
| `/api/ik/mouse-aim/heartbeat` | Session ID, optional execution ID and replacement plan ID | Renews the five-second lease; separately acknowledges the received execution or replacement response. |
| `/api/ik/mouse-aim/stop` | Session ID | Invalidates the session/plan and stops its owned running path. |

All endpoints use POST. Global Stop and Disconnect remain available. Session
expiry is checked by the controller poller even during planning or idle
tracking. The first motor command waits for the browser to receive and
acknowledge the execution response. A session heartbeat alone cannot authorize
a pending execution. A lost response therefore cannot silently start motion.
If Stop cannot be delivered, heartbeat loss expires execution. Software Stop
depends on the network and is not a physical emergency stop.

## Verification and limits

Tests use fake robot transport and mocked browser hardware endpoints. Browser
predicted paths come from the real simulation planner. Tests cover gates, live
starts, accepted cursor goals, retained frames, one-use plans, ownership,
stale responses, acknowledgement, expiry, collision rejection, Stop,
disconnect, latest-cursor coalescing, live display, velocity/acceleration
bounds, servo deadband, mode verification/restoration and moving-target
handoffs. No physical motor position command was sent by the agent during
automated verification; physical calibration and smoothness of the updated
controller remain to be verified on the arm.

The [v2 pinned-policy smoke result](../output/hardware_mouse_aim/v2_mouse_actual_policy_validation.json)
uses actual checkpoint inference and checked C2 execution with an in-memory
robot. The local shortcut is deliberately bypassed to exercise the neural
fallback; two chunks reach its stationary target. Lifecycle tests additionally
cover C2 joins, slow-motor pacing, live drift, preview/curve consistency and
unreadable telemetry. These are offline checks, not physical performance data.

The first velocity-controller hardware trial stopped after two samples. Pi
logs showed each motor command blocking for 512–513 ms waiting for an absent
acknowledgement, followed by combined duplicate GET_ANGLES packets. This
exceeded the original 500 ms controller cutoff before an effective position
reference was sent. The transport fix removes that Pi wait, reconstructs
packet boundaries on both ends, and retains a two-second stale-cycle Stop
limit. Tests include split/coalesced packets, the real SDK's encoder/decoder,
a 550 ms delayed reply, bounded tracking under delayed telemetry and Stop on
two-second delay. The captured failure is
[archived here](../output/hardware_mouse_aim/no_motion_velocity_failure.json).

The first physical user trial recorded 65 commands and a first-waypoint settling
timeout. Telemetry samples retain the command, reference velocity, active plan
and motion status for diagnosis. Timeout messages include the actual route
joint error. The earlier
[fake-transport comparison](../output/hardware_mouse_aim/smoothing_comparison.json)
describes the preceding waypoint scheduler, not this velocity feedback loop.
See [current validation](../output/hardware_mouse_aim/transport_validation.json)
for the tested implementation and read-only bridge verification.

Classical mode checks modeled arm self-collision; v2 also checks the pinned table.
External obstacles, added tools
and cables, dynamics and continuous collision detection are not included.
Intermediate motion may change tool-Z direction. The existing operator
frame/path checks remain necessary for physical use.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s webui/tests_python -q
cd webui
npx playwright test tests/v2-mouse.spec.js tests/mouse-aim-hardware.spec.js tests/mouse-aim.spec.js tests/hardware-diffusion.spec.js tests/hardware-ik.spec.js
npm run build
```
