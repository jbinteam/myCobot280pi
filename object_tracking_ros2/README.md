# Standalone ROS 2 arm control and object following

## C++ ROS node

The default controller is the compiled `object_tracking_cpp/object_follow_cpp`
executable, named `/object_follow`. It uses `rclcpp` for ROS subscriptions,
services, timers, exact-stamp TF lookups, detection validation, measured feedback
publication, joint states, and target publication. ROS interfaces and RViz topics
are compatible with the previous node.

The C++ executable embeds CPython with pybind11 to reuse the checked learned-IK
and RRT* planner, tracking/deadband/approach state machine, general joint/pose
commands, collision checks, and encoder executor. These components remain
Python. The direct backend lives in the same process and owns one Pi connection;
it does not launch a Python ROS node or WebUI server. This frontend port does not
establish a planner or end-to-end speedup.

```bash
# Default full system: compiled C++ ROS frontend.
./start_ros_system.sh
# Explicit C++ selection, keeping the configured distance.
./start_ros_system.sh controller_language:=cpp tracking_mode:=keep_distance
# Select the previous Python ROS frontend.
./start_ros_system.sh controller_language:=python
# Disarmed controller-only dry run (no automatic robot connection).
./start_object_tracking.sh --dry-run controller_language:=cpp
```

Both launchers build and install `object_tracking_ros2` and `object_tracking_cpp`
before starting. Build dependencies are a C++17 compiler, CMake, ROS Jazzy's
`ament_cmake`, `rclcpp`, TF/message packages, `pybind11-dev`, and Python 3.12
development headers. The existing project `.venv` supplies planner/SDK packages.
The source checkout is required and passed via the `project_root` parameter.
`/object_follow/status` and `/object_follow/timing` identify the native frontend
with `"implementation": "cpp"`.

Stop/SIGINT cancels the session, delivers the checked Stop/disconnect sequence,
and waits for Python workers before destroying the embedded interpreter.
The native node is validated with synthetic detections and simulated feedback;
physical tracking speed and hardware behavior still need a live run.

## Camera-aware approach planning

Both ROS frontends now support the WebUI's strict-first camera-FOV fallback.
With `tracking_mode:=approaching`, each request first tries pointing tool Z
directly at the object. Only `ik_failed` or `no_path` without a playable strict
continuation permits a second search with orientation free inside the pinned
rectangular wrist-camera view. A playable strict prefix is retained. Both passes
use the same measured start, motion boundary, frozen disk and camera mount.
Collision, occlusion, complete-route visibility, timing, encoder rounding and
measured arrival checks still apply. Invalid starts and visibility failures
never trigger relaxation. Search exhaustion is not proof of unreachability.
Stop, detector-loss pause and disconnect cancel unfinished numerical searches
and sampled route checks before another orientation pass can begin.

The startup parameter `approach_camera_fov_fallback` defaults to `true` and
applies only to approaching; `keep_distance` retains strict pointing. Disable
the fallback for strict-only comparison:

```bash
./start_ros_system.sh tracking_mode:=approaching approach_camera_fov_fallback:=false
# Controller-only, disconnected dry run with the updated default policy:
./start_object_tracking.sh --dry-run approach_camera_fov_fallback:=true
```

The policy is pinned when the controller starts. Restart with a new launch
argument to change it. The optional HTTP backend must acknowledge the updated
orientation protocol before tracking can arm; an older server is rejected.
`/object_follow/status` includes `orientation_policy`, the most recent
`orientation_selection` (strict/fallback outcomes and processing times), the
pinned `visibility_camera`, and the active plan's `gaze.orientation_mode`.
Arrival in a relaxed plan requires the same disk and camera image bounds,
rather than strict tool pointing. Every later stage tries strict planning again.

RViz now draws the camera frustum from measured joints and the controller's
configured optical mount, plus a target ray and image-bounds readout. Its state
label identifies an active FOV fallback. The C++ frontend forwards the configured
horizontal/vertical view angles and margin/near plane to the same model as the
Python frontend (default **82 × 68 degrees**). The view is modeled, not camera
video; image-bounds markers alone do not indicate freedom from occlusion.
Toggle these drawings independently of arm control:

```bash
ros2 param set /arm_tracking_visualization show_camera_fov false
ros2 param set /arm_tracking_visualization show_camera_fov true
```

The existing 20 mm approach stages, final-distance hold, detector-loss
pause/resume and live RRT* display remain available. ROS approaches use the
configured initial distance (`cone_distance_mm`, default 350 mm). The WebUI's
random-object repeat/reset belongs to simulation; ROS continues following real
accepted detections at the final distance. This upgrade has been exercised with
simulated motors; camera calibration and physical qualification remain deferred.

## Tracking timing diagnostics

The controller publishes `/object_follow/timing` once per second. Each stage
contains a lifetime count and last/median/p95/max milliseconds over its latest
512 samples. The diagnostic does not change the motion policy or speed limits.

After starting the system, collect a read-only 60-second recording:

```bash
cd /home/jb/codex/projects/myCobot280pi
source /opt/ros/jazzy/setup.bash
python3 scripts/collect_tracking_timing.py --seconds 60 \
  --output output/tracking_timing_live.json
# Or inspect the live topic:
ros2 topic echo /object_follow/timing
```

`detection.capture_to_callback` measures camera capture to receipt of an
accepted detector pose, including detection and ROS delivery.
`detection.callback_to_target` includes waiting for measured arm feedback/TF
and the next processing tick. `detection.capture_to_plan_start` and
`detection.capture_to_execute_return` expose age at planning and executor
acceptance; the latter does **not** measure physical arm response.
`backend.*` includes backend call time and lock waits. `planner.*` reports the
planner's existing internal timers. `ros.tick` is callback work;
`ros.tick_interval` is the interval between callbacks. Capture ages require
synchronized clocks. Nested stages overlap and must not be added.

`hardware_prepare.*` exposes the hardware adapter's existing measurements:
`live_feedback_read` includes the initial controller-lock wait and state
capture; `validation_and_encoder_preparation` includes post-planner validation,
any profile rebuilding, final controller-lock wait and feedback recheck.
These broad stages are not pure serial-I/O or CPU timings. `total` spans the
adapter preparation and overlaps these stages and `backend.ik.mouse-aim.plan`.

`backend_latest_seconds` contains the latest observed executor timing fields,
including telemetry cycle and send interval. These are snapshots from status
polling, not a complete recording of every hardware command. Empty fields mean
no corresponding backend sample is available. The collector never connects,
arms, or commands the robot. Start the updated controller through the normal
launcher to build and load these changes.

An offline benchmark runs the actual learned-IK/classical tracking planner
without opening hardware:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/profile_object_tracking.py \
  --repeats 10 --output output/tracking_latency_run
```

The JSON report separates unprofiled wall measurements from instrumented
hotspot profiles. Five diagnostic targets cover holding, retargeting, and an
approach step. They do not characterize the whole workspace or establish a C++
speedup. Camera, ROS delivery, and hardware latency require a live recording.

The ROS 2 Jazzy node `/object_follow` owns the Pi connection and runs the arm
controller, planner, collision validation and encoder executor in its own
process. The default `direct` backend requires **no WebUI, HTTP server or browser**.
It uses the selected learned IK + classical RRT* planner and the shared,
feedback-paced encoder velocity profile with 5% rise / 10% fall. General joint
and tool-pose commands and detected-object following share exclusive ownership.

The WebUI remains a separate development sandbox. Some shared implementation
modules still live under `webui/`; importing those libraries does not start or
import the HTTP server. The ROS package currently requires this project checkout,
its meshes, pinned learned model artifacts and the project's Python environment;
it is not a self-contained binary distribution.

## Dependencies and startup

### One-command full system with RViz

After a fresh robot boot, start the TCP bridge **on the arm Pi** first:

```bash
python3 -u ~/mouse_control_server/pi_server.py 192.168.77.225
```

Alternatively copy `start_pi_bridge.sh` from this project to the Pi and run it
there. The PC launcher does not require SSH credentials and does not start a
second Pi server. Power the arm and connect its network before launching.

On the PC, in a desktop terminal:

```bash
cd /home/jb/codex/projects/myCobot280pi
./start_ros_system.sh
```

This sources ROS, builds/starts the standalone controller, starts the existing
Orbbec + detector pipeline, and opens RViz. **Tracking arms automatically once**
fresh measured feedback is available. With no accepted detection, it stays
armed and idle; RViz shows **ARMED:WAITING**. Put the trained object in view
to begin following automatically. Losing the object returns to the waiting
state without disarming. A subsequent explicit Stop is never automatically
overridden. If initial arming fails or you stop tracking, run `./arm_tracking.sh`
when the target is ready.

RViz shows the measured arm meshes, detected target, 350 mm / 30° cone and disk,
camera/target frames, object point cloud, camera detection image, and arm state.
The visualization is read-only and does not publish motion commands.

Live RRT* visualization uses the same real planner observer as the WebUI:
thick magenta branches show the growing/rewiring joint-space tree projected to
tool positions, and a thicker blue line shows the checked accepted route.
The orange sphere is the detected target. Direct candidates are labeled;
IK must supply an endpoint before RRT* can route to it. Search drawings never
command the arm. Both C++ and Python controllers support direct and HTTP backends.

Choose at startup with `./start_ros_system.sh --show-rrt true` or
`--show-rrt false` (default true). The ROS launch equivalent is
`show_rrt:=true` / `show_rrt:=false`; standalone `start_object_tracking.sh`
accepts the same launch argument. These normal system launchers can connect
hardware; simulation verification uses injected fake motors instead.
`--no-rviz` still controls whether the entire RViz window opens.

Toggle while running without changing tracking or motion ownership:
```bash
ros2 param set /object_follow show_rrt false
ros2 param set /object_follow show_rrt true
```
Turning it off clears the drawings and invalidates late diagnostics; turning
it on shows the next planning request (it does not restart an active search).
For a display-only toggle that keeps collecting diagnostics, check/uncheck
**RRT* search and checked route** in RViz's Displays panel. It subscribes to
`/arm/planning_visualization` separately from the arm/object/region display.
Alternatively, set `show_rrt` on `/arm_tracking_visualization`; individual
`rrt_search`, `rrt_route` and `rrt_status` namespaces can also be hidden.
Magenta `#ff40cf` and blue `#4080ff` use brighter shades for the dark RViz scene.
Diagnostics are bounded to 4,096 nodes per tree and 16 cached attempts, sent
at most 10 Hz on `/object_follow/planning_visualization`. RViz line widths are
4 mm / 6 mm in world space, following the
[ROS Marker line scale convention](https://docs.ros.org/en/jazzy/Tutorials/Intermediate/RViz/Marker-Display-types/Marker-Display-types.html).
Only current-session, fresh-target proposals that pass curve/encoder gates
are highlighted. LOST/pause/Stop clear diagnostics; measured-arm markers remain.

The acceptable tracking area is the translucent circular plane, perpendicular
to the object-to-camera unit vector captured for the accepted target. Its center
is `object + distance * unit_vector`; its radius is `distance * tan(half_angle)`.
The plane remains fixed for that target, with the configured axial allowance.
The cone lines show its construction; the whole cone volume is not acceptable.
The default camera origin coincides with the EEF origin. Tool Z must continue
to point at the object from whichever checked position is selected on the disk
during strict planning. An approaching FOV fallback may relax that orientation
while retaining the same disk and the pinned camera's image bounds.

The image displays separate segmentation from pose estimation:
`Segmentation - 2D detection` shows `/object_pose/detection_image`,
`Segmentation - object mask` shows `/object/mask`, and
`Pose estimation - 3D box` shows `/object_pose/box_image`.
The 3D box disappears when pose estimation is rejected, even if segmentation
still detects the object. Tracking consumes accepted `/object_pose/filtered`
poses, so a visible segmentation mask alone does not enable following.

```bash
./start_ros_system.sh --no-arm    # Connect, detect and visualize without motion
./arm_tracking.sh                # Arm later from another terminal
./stop_ros_system.sh             # Stop motion and all processes owned by this launcher
./start_ros_system.sh --restart  # Cleanly restart a previous managed run
```

Ctrl+C in the launcher terminal also stops the full system, including RViz.
Existing separately launched detector/controller processes must be stopped
first; this launcher refuses to start competing owners. Logs are saved under
`output/ros_system/latest/`. Use `--no-rviz` for a headless session, or
`--robot-host`, `--scanner-root`, `--dataset`, and `--category` to override the
current setup. Defaults use `my_object_20260925_r2`, category `my_object`, Orbbec
at 30 Hz, and the arm at `192.168.77.225:9000`.

### Configurable full-system ROS launch

The full-system launch starts the detector/camera, direct arm controller,
read-only visualization, and RViz through the existing owned-process supervisor.
The Pi TCP bridge must already be running. From the project root, after building
(the shell launcher builds automatically):

```bash
source /opt/ros/jazzy/setup.bash
source output/object_follow_ros2/install/setup.bash

# Keep a fixed cone distance.
ros2 launch object_tracking_ros2 full_system.launch.py \
  tracking_mode:=keep_distance cone_distance_mm:=350 cone_half_angle_deg:=30

# Approach in stages, then keep tracking at the final distance.
ros2 launch object_tracking_ros2 full_system.launch.py \
  tracking_mode:=approaching cone_distance_mm:=350 cone_half_angle_deg:=30 \
  approach_final_distance_mm:=150 approach_step_mm:=20 \
  approach_distance_tolerance_mm:=10 sdk_speed:=70 target_deadband_mm:=30
```

The shell shortcut accepts the same ROS arguments:

```bash
./start_ros_system.sh tracking_mode:=keep_distance cone_distance_mm:=250
./start_ros_system.sh tracking_mode:=approaching approach_final_distance_mm:=150
```

Existing flags also work, for example
`./start_ros_system.sh --tracking-mode keep_distance --cone-distance-mm 250 --no-arm`.
Use either ROS `name:=value` arguments or `--flag` arguments in one invocation.

| Launch argument | Default | Meaning |
|---|---|---|
| `controller_language` | `cpp` | C++ ROS frontend or previous `python` frontend |
| `tracking_mode` | `approaching` | `keep_distance` or `approaching` |
| `cone_distance_mm` | `350` | Fixed distance or initial approach distance, axial mm |
| `cone_half_angle_deg` | `30` | Cone **half-angle**, degrees |
| `approach_final_distance_mm` | `150` | Final axial approach distance |
| `approach_step_mm` | `20` | Reduction per accepted stage |
| `approach_distance_tolerance_mm` | `10` | Plus/minus axial allowance; at most half the step |
| `approach_camera_fov_fallback` | `true` | Strict-first camera-FOV fallback for approaching; startup setting |
| `target_deadband_mm` | `30` | Object-position jitter sphere radius |
| `sdk_speed` | `70` | SDK speed; allowed range 1–80 |
| `reference_speed_rad_s` | `1.0` | Requested joint-vector speed, capped by SDK scaling |
| `auto_arm` | `true` | Arm once with fresh feedback and wait for detections |
| `rviz` | `true` | Open RViz |
| `detector` | `true` | Start detector; false consumes existing ROS detection topics |
| `camera_backend` | `orbbec` | `orbbec`, `realsense`, or `auto` |
| `camera_rate` | `30` | Requested camera frame rate |
| `dataset` / `category` | `my_object_20260925_r2` / `my_object` | Existing detector artifacts |
| `robot_host` / `robot_port` | `192.168.77.225` / `9000` | Pi bridge endpoint |
| `project_root` / `scanner_root` | Current checkout / sibling `objectScanner` | Runtime locations |
| `restart` | `false` | Stop the prior managed run before starting |
| `print_config` | `false` | Print resolved configuration without connecting or launching hardware |

```bash
ros2 launch object_tracking_ros2 full_system.launch.py --show-args
ros2 launch object_tracking_ros2 full_system.launch.py print_config:=true \
  tracking_mode:=keep_distance cone_distance_mm:=250
```

Invalid mode, distance, angle, speed, or allowance values fail before any child
process starts. `auto_arm:=false` connects and visualizes with motion disabled;
run `./arm_tracking.sh` later. Ctrl+C or `./stop_ros_system.sh` stops the managed
system. Launch arguments override the corresponding controller YAML values;
YAML remains the defaults for controller-only startup and the camera mounting.

### Controller-only startup

Use ROS 2 Jazzy, `colcon`, the ROS message/TF dependencies in `package.xml`, and
this checkout's `.venv` with its existing NumPy, SciPy, PyTorch, PyBullet,
collision libraries and pymycobot installation. The launcher retains the sourced
ROS Python paths. The Python frontend runs with `.venv/bin/python`; the C++
frontend embeds Python and loads packages from that environment. The trained model
artifacts and geometry must remain at their repository paths. It performs an
incremental package build before launching so code changes are installed.

Only one process may own the Pi serial bridge. Stop the existing WebUI tracking
session and **disconnect the WebUI from the arm** before starting direct hardware
control. The Pi bridge on port 9000 must already be running. Starting ROS connects
and reads feedback; it does not enable motion.

From the repository root:

```bash
./start_object_tracking.sh --hardware
# Optional address override:
./start_object_tracking.sh --hardware robot_host:=192.168.77.225 robot_port:=9000
```

Both hardware and dry-run modes start **disarmed**. `--hardware` enables the
checked hardware path, marks the configured flange/table frame assumptions
verified, and requests an initial Pi connection. The default command (or
`--dry-run`) transforms and reports detections without sending motion; it does
not auto-connect. A dry run still needs measured arm feedback to transform
camera-frame targets, so connect explicitly when that feedback is required.

In another terminal:

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic echo /object_follow/status
ros2 topic echo /joint_states
# Connect/reconnect explicitly if needed:
ros2 service call /object_follow/connect std_srvs/srv/Trigger '{}'
```

A custom camera/configuration file can be selected with
`config:=/absolute/path/object_follow.yaml`. Direct control is the deployment
path. The older HTTP adapter is retained for sandbox compatibility and is chosen
explicitly with `backend:=http auto_connect:=false`; it requires the separate
local hardware WebUI server.

## General arm commands

To request all six joints at zero degrees from the running ROS system:

```bash
./zero_arm.sh
```

This stops tracking, waits for Stop to finish, enables checked joint motion,
and publishes a fresh zero-joint goal. It exits after the controller acknowledges
the request; planning/execution may still be running. Check `arm_motion` in
`/object_follow/status` for completion or rejection. Tracking remains disabled.

General commands do not need an object detector. Enable them only after fresh
measured feedback is available:

```bash
ros2 service call /arm/enable_motion std_srvs/srv/Trigger '{}'
```

Publish a single `sensor_msgs/msg/JointState` on `/arm/command_joints`, containing
**all six joint names** from `/joint_states` and six positions in **radians**.
Names identify the joints; no partial-joint command is accepted. Leave velocity
and effort arrays empty. Both joint and pose messages require a fresh, nonzero
wall-clock header timestamp (within the configured 0.75-second age bound); a
default zero timestamp is rejected. The controller
plans and checks the route from the current measured pose before executing.

Publish a `geometry_msgs/msg/PoseStamped` on `/arm/command_pose` with
`header.frame_id: g_base`, position in **metres**, and a valid unit quaternion in
`[x,y,z,w]` order. This is a **five-dimensional pose task: XYZ plus tool-Z pointing
direction**. Quaternion roll about tool Z is free; it is not a full six-axis
orientation constraint. Both command interfaces use the same selected planner,
profile and checked encoder executor. Execution results/errors appear under `arm_motion` in `/object_follow/status`;
input-validation failures are logged by the ROS node. Only one general command
can run at a time; wait for completion or Stop it before issuing the next goal.

Object following and general commands cannot be enabled simultaneously. Stop
the active mode before enabling the other. Stop and disarm either mode with:

```bash
ros2 service call /arm/stop std_srvs/srv/Trigger '{}'
```

Disconnect (which also stops motion) with:

```bash
ros2 service call /object_follow/disconnect std_srvs/srv/Trigger '{}'
```

## Detected-object following

Start the existing `objectScanner` pipeline separately:

```bash
bash /home/jb/codex/projects/objectScanner/scripts/collect_process/07_run_live.sh \
  my_object_20260925_r2 my_object camera_backend:=orbbec camera_rate:=30 rviz:=false
```

The node uses the accepted **3D box center** from `/object_pose/filtered` in place
of the mouse cursor. Once status reports fresh measured feedback, arm following
even if no object is visible yet:

```bash
ros2 service call /object_follow/arm std_srvs/srv/Trigger '{}'
```

The response is `Armed; waiting for object` when no fresh target exists. Motion
starts only after detector state `TRACKING` and a fresh transformed target arrive.

Approach mode is enabled by default. It begins at **350 mm axial distance /
30° cone half-angle**, giving an acceptable disk radius of approximately
**202.1 mm**. After each checked trajectory finishes and measured feedback
confirms acceptance of the current region within a **±10 mm axial allowance**,
the next distance is reduced by
**20 mm**, down to **150 mm**. The final disk radius is approximately **86.6 mm**.
The cone angle stays fixed, so `radius = distance * tan(30°)` shrinks the area
as the arm approaches. Each new distance stage captures a fresh centerline from
the measured camera/EEF position to the object, then regenerates the cone and
perpendicular disk along that line. The disk is fixed within that stage, including
planning retries. RViz adopts it when the checked stage trajectory becomes active.
These are axial cone-plane offsets; an off-center point
on the final disk can have a larger Euclidean object distance. The distance
allowance gives a final axial band of **140–160 mm** around the 150 mm plane.
The executor can stop and settle inside this band before the exact trajectory
endpoint; cone radius, pointing, collision and measured-feedback checks still
apply. The allowance does not replace the separate 30 mm target-jitter deadband.

Before IK or motion search, the shared planners apply a conservative physical
reach precheck derived from the URDF chain. A goal point, or an entire cone disk
including its planning allowance, outside the enclosing flange workspace returns
`outside_workspace` with no trajectory. Individual outside-bound disk samples are
skipped. The check applies to the **EEF goal region**, not the object position;
an object beyond arm reach can still be observed from a reachable stand-off pose.
Overlap only means the goal has not been geometrically excluded. Orientation,
operating joint limits and collision feasibility still require IK and checked
planning; exhausting their finite search budget does not prove unreachability.

The shared learned-IK/RRT controller now selects among visible cone-region
endpoints with bounded candidate improvement. It checks the full geometric route,
the timed q/v/a continuation, rounded encoder targets and each live
measured-to-command connection. Initially invisible objects produce no playable
route. The cone disk/slab is an endpoint constraint, not a trajectory corridor.
This source integration has fake-motor tests; it has not been deployed or
qualified on physical hardware. Visibility uses the nominal **82° × 68° RGB
field of view** of the Orbbec Gemini 305's preferred 640 × 480 (4:3) color profile
([datasheet, table 4-7-3](https://mm.digikey.com/Volume0/opasdata/d220001/medias/docus/8888/Orbbec_Gemini-305-Datasheet.pdf)).
`horizontal_fov_deg` and `vertical_fov_deg` in the controller YAML set the same
model for the simulation server and ROS controller. A 1280 × 800 / 848 × 530
color stream instead has nominal 94° × 68° FOV; 1280 × 720 / 848 × 480 is 94° × 62°.
Actual rectified stream intrinsics and edge coverage remain unvalidated.
Visibility checks a point object, with the configured ROS flange-to-optical mount pinned
through planning and execution. The current mount is identity and explicitly
assumed. ROS arming rejects a controller camera configuration mismatch. See the
[controller integration details](../simulation/diffusion/README.md#measured-feedback-controller-integration)
for budgets, validation and remaining camera work.

At the final distance the arm continues following the object. LOST detections
pause approach at its current stage; reacquisition resumes it. Unreachable or
collision-rejected stages retain the last checked position and retry; they do
not authorize a further distance reduction. Explicit Stop/re-arm starts again
at 350 mm. A changed object target is tracked at the current distance before
approaching further. RViz renders the active checked cone and displays
`APPROACHING:<distance>mm` or `HOLDING:150mm`.

For full-system startup, configure the launch arguments above. For controller-only
startup, configure `approach_enabled`, `approach_final_distance_mm`,
`approach_step_mm`, and `approach_distance_tolerance_mm` in
`config/object_follow.yaml`, then restart the controller. The distance tolerance
must be positive and no greater than half a stage step.
Set `approach_enabled: false` for fixed 350 mm tracking.

The disk is perpendicular
to the arm-to-target axis and tool Z aims toward the detected object. Targets use
a **30 mm radius spherical deadband in robot-base coordinates**: coordinates
change only after movement exceeds 30 mm from the held center (not from the
previous raw detection). Fresh detections inside or exactly on the sphere still
refresh the measurement timestamp and keep the current target/trajectory;
their jitter does not trigger replanning. Configure
`target_deadband_mm` in the YAML.

Detection `LOST` or an expired accepted detection **stops motion and holds the
current pose while keeping following armed and its session leased**. Fresh
accepted detections resume with a newly checked plan from measured joints.
Status reports `paused` and `pause_pending`. An explicit Stop, Ctrl+C, a hardware
fault or invalid control ownership disarms; those conditions require re-arming.

```bash
ros2 service call /object_follow/stop std_srvs/srv/Trigger '{}'
```

## Camera and measured frames

The detector's `PoseStamped.pose.position` is the calibrated `box_center`, in
**metres**, in `camera_color_optical_frame`. It is not a bounding-box pixel or
a point-cloud surface centroid. Object orientation is ignored for following.

The supplied task assumption places the **camera optical origin at the EEF**,
with optical Z aligned to tool Z and zero translation. It is not a measured
hand-eye calibration. `camera_link` and an optical frame have different axes;
identity is applied directly to the optical frame. For a calibrated mount, edit
`eef_to_optical_translation_m` and `eef_to_optical_quaternion_xyzw` in the YAML to
describe `T_joint6_flange_camera_color_optical_frame`, and set the mount flags to
reflect the actual calibration. Quaternion order is `[x,y,z,w]`.

`simulation/camera_model.py` shares that mount with the planner, independent
curve/encoder checks and live execution. Direct Python and C++ ROS runtimes pass
their startup mount settings into the controller. The optional HTTP controller
loads `config/object_follow.yaml` when starting a tracking session; the ROS
follower compares the returned camera configuration before arming. A mismatch
releases the new session and sends no motion. Mount changes require a new
session; retargets cannot replace the pinned camera. Standalone region planning
uses the same YAML defaults. The FOV remains assumed and `calibrated` remains
false even if the mount itself is marked calibrated.

The controller computes FK from measured joint angles, stamped when each angle
reply arrives. ROS publishes `/joint_states`, `/object_follow/eef_pose` and the
`g_base -> joint6_flange` transform, plus the configured static optical mount.
Each detection is transformed at its own timestamp:

```text
p_base = T_base_eef(t_detection) * T_eef_optical * p_optical
```

Buffered detections wait for measured feedback to bracket their timestamps. The
node does not substitute the newest transform for an older camera measurement.
Timestamps are host receipt times; neither camera exposure time nor the robot's
internal sampling time is available through these transports. The default
accepted detection age bound is 0.75 seconds.

The direct feedback cache can be read while serial/control work is underway.
Validated angles enter it immediately, before motion-status and health queries
complete. Reading the cache does not refresh measurement timestamps. Streaming
queries retain a 0.5-second total budget across retries and health checks; an
actual hardware-query failure stops motion.

## Motion behavior

Only accepted fresh filtered poses with `TRACKING` state authorize tracking
motion. Invalid frames, nonfinite coordinates, nonpositive depth and future
stamps cannot authorize motion. ROS inputs use metres/radians; conversion to the
planner's millimetres and hardware degrees occurs inside the controller.

One planner runs at a time. Checked plans are acknowledged before encoder
execution. Moving tracking replacements retain their execution owner, and stale
joins retry from live joints. Partial-path completion and gaze misses can
request a correction for a stationary object. Independent lease work and
asynchronous arming keep planning and hardware startup from blocking detection
callbacks; Stop invalidates an in-flight start or plan.

Defaults are **SDK speed 70** on the 1–100 scale, requested reference speed
**1.0 rad/s**, and **5% rise / 10% fall**. The existing executor applies the lower
of the reference request and `radians(20 * sdk_speed / 30)`, giving an effective
**0.81449 rad/s (46.67°/s) joint-velocity-vector limit** at SDK 70. This is a
configured trajectory limit, not a measured percentage of the arm's maximum
physical performance. General joint/pose commands also use SDK speed 70.
The global SDK ceiling remains 80. Feedback paces motion when motors
lag. Self/table collision checks apply to planned paths, measured poses and
bounded measured-to-command segments. Moving external obstacles are outside
this task's assumptions. Added camera geometry is not automatically included
in the arm collision model.

## ROS interfaces

| Interface | Type | Contract |
|---|---|---|
| `/object_pose/filtered` | `geometry_msgs/PoseStamped` | Accepted camera-frame box center, metres |
| `/object_pose/tracking_state` | `std_msgs/String` | `TRACKING` / loss gate |
| `/object_follow/target` | `geometry_msgs/PointStamped` | Held transformed center, metres in `g_base` |
| `/joint_states` | `sensor_msgs/JointState` | Measured six-joint positions, radians |
| `/object_follow/eef_pose` | `geometry_msgs/PoseStamped` | Measured flange pose, metres in `g_base` |
| `/object_follow/status` | `std_msgs/String` | JSON feedback, mode, target age, ownership, approach stage and command diagnostics |
| `/object_follow/connect` | `std_srvs/Trigger` | Connect configured Pi transport, no motion enable |
| `/object_follow/disconnect` | `std_srvs/Trigger` | Stop/disarm and close this controller's transport |
| `/object_follow/arm` | `std_srvs/Trigger` | Arm tracking; wait idle until a fresh transformed detection arrives |
| `/object_follow/stop` | `std_srvs/Trigger` | Stop/disarm tracking and general commands |
| `/arm/enable_motion` | `std_srvs/Trigger` | Enable general commands with fresh feedback |
| `/arm/command_joints` | `sensor_msgs/JointState` | All six named joint targets, radians |
| `/arm/command_pose` | `geometry_msgs/PoseStamped` | `g_base` XYZ metres + quaternion tool-Z direction, roll free |
| `/arm/stop` | `std_srvs/Trigger` | Stop/disarm all motion owned by this ROS node |

## Validation without a physical arm

ROS-only checks use the sourced system Python. Direct hardware lifecycle checks
use the project environment with system pytest made available:

```bash
source /opt/ros/jazzy/setup.bash
PYTHONPATH="$PWD/object_tracking_ros2:$PYTHONPATH" python3 -m pytest -q \
  object_tracking_ros2/test/test_core.py object_tracking_ros2/test/test_ros.py
PYTHONPATH="$PWD/object_tracking_ros2:$PWD/pymycobot:$PWD" .venv/bin/python -c \
  "import sys; sys.path.append('/usr/lib/python3/dist-packages'); import pytest; raise SystemExit(pytest.main(['-q','object_tracking_ros2/test/test_direct_lifecycle.py']))"
```

`test_camera_fallback.py` runs the saved strict-IK failure through the actual
ROS session, direct planner, curve/encoder validation and executor using only
in-memory motors. It checks the strict-first handshake, rejects unauthorized
relaxation, and verifies relaxed measured arrival while preserving the disk and
rectangular image bounds. `test_visualization.py` checks frustum geometry with a
translated/rotated optical mount and independent display toggling.

`test_direct_lifecycle.py` uses the real direct controller, collision validation,
encoder conversion and executor with an in-memory robot. Tracking tests use
deterministic proposals; joint and pose command tests run the real classical
planner, including learned IK for pose commands. Network connection, bind and
HTTP calls are disabled. It
checks moving retargeting, LOST beyond the session lease interval, recovery from
measured joints, stale-detection pause, stale-join retry, Stop during planning,
hardware-fault disarming, exact six-joint endpoints, and XYZ/tool-Z pose
endpoints. It does not validate physical robot accuracy or object detector
quality.

After a bounded `no_path` or `ik_failed` response, the ROS follower advances
the planner seed for the next attempt. This avoids repeating an identical
exhausted search from a held pose. Each attempt retains its existing budgets,
frozen region and collision/visibility/encoder checks. A successful execution,
new held target or new armed session resets the seed to 42. Search exhaustion
still does not certify that the region is unreachable.

`test_full_pipeline.py` connects the built native C++ ROS frontend to the actual
local HTTP handler, controller, learned IK/RRT planner and encoder executor.
Only the motor transport is replaced by an instantaneous in-memory encoder
plant. Synthetic optical detections use the configured camera transform and
the exact timestamp of measured fake-joint feedback. This exercises the full
350→150 mm staged approach at the configured SDK speed, LOST/resume, detection
expiry, deadband, retargeting, Stop and frame faults across the ROS/API boundary.
It opens an ephemeral localhost HTTP listener and isolated ROS domain 185;
it never connects to the Pi. A built native executable is required.

```bash
source /opt/ros/jazzy/setup.bash
OPENBLAS_NUM_THREADS=1 \
PYTHONPATH="$PWD/object_tracking_ros2:$PWD/pymycobot:$PWD/webui/tests_python:$PWD/webui:$PWD:$PYTHONPATH" \
  .venv/bin/python -c \
  "import sys; sys.path.append('/usr/lib/python3/dist-packages'); import pytest; raise SystemExit(pytest.main(['-v','object_tracking_ros2/test/test_full_pipeline.py','--basetemp=output/full_pipeline_test']))"
```

The test writes `approach_trace.json` for independent replay with
`scripts/audit_visible_approach.py`, plus `pipeline_trace.json` containing optical
detections, base targets, joint feedback, encoder calls and lifecycle events.
Use a new output directory to preserve previous runs. These checks do not
qualify camera inference, optical calibration, full-object visibility, motor
dynamics or physical stopping distance.
