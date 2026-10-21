# Project handover and hands-on guide

Prepared 2026-10-21 (Asia/Bangkok). This is a handover of a research prototype
with a simulation-tested controller. It is not physical deployment acceptance.
The earlier session conversation was unavailable; historical results below
come from the saved project records and inspected source.

## 1. What you are taking over

The system turns accepted RGB-D object detections into checked robot motion.
The default route uses learned IK proposals, numerical refinement, a direct
joint path or RRT* detour, checked timing, rounded encoder validation, and
feedback-paced execution. Diffusion v2 remains an alternative experiment.
The C++ ROS frontend embeds Python; the planner/executor have not been ported
to C++ and no end-to-end speedup is established by that frontend change.

```text
External objectScanner → stamped optical-frame box center + TRACKING state
  → ROS C++ frontend (or Python alternative)
  → timestamped measured FK + camera mount → target in g_base
  → deadband / approach stages / strict-first camera policy
  → learned IK + numerical refinement + RRT* + collision/timing checks
  → encoder validation + measured-feedback execution → Pi TCP/serial bridge

WebUI → same planning/control components, or explicit simulation-only mode
```

| Location | Responsibility |
| --- | --- |
| `webui/src/`, `webui/server.py` | Browser interaction, HTTP API and session ownership |
| `webui/hardware_ik.py`, `webui/v2_mouse.py`, `webui/encoder_*.py` | Checked execution, tracking and encoder handling |
| `simulation/kinematics.py`, `joint_limits.json` | URDF FK and firmware-constrained operating limits |
| `simulation/diffusion/classical_service.py`, `ik_seed.py` | Active learned IK + classical planning |
| `simulation/diffusion/` | Shared planning, diffusion, scene and validation routines |
| `simulation/camera_model.py` | Pinned optical mount and visibility model |
| `object_tracking_ros2/object_tracking_ros2/` | Tracking state machine, direct backend, commands, visualization |
| `object_tracking_cpp/` | Native ROS frontend and embedded Python integration |
| `object_tracking_ros2/config/object_follow.yaml` | Robot address, frames, camera and controller defaults |
| `hardware/pi_server.py`, `pi_transport.py` | Framed TCP/serial bridge on the Pi |
| `scripts/ros_system.py`, root `*.sh` | Launching, diagnostics and shutdown |
| `webui/public/models/` | Runtime URDF/meshes, upstream license and provenance |
| `simulation/artifacts/` | Runtime models/manifests plus separate historical training material |

## 2. Current behavior and constraints to preserve

- ROS inputs: positions in metres, joints in radians. WebUI target coordinates
  use millimetres and hardware joint commands use degrees. Pose commands control
  XYZ plus tool-Z direction; rotation about tool Z is free.
- Optical X is right, Y down, Z forward. Transform detections at their own
  timestamps using measured joints; do not substitute the newest transform.
- Accepted targets require fresh detections and `TRACKING`. The target deadband
  is a 30 mm sphere around the held target, not the preceding raw sample.
- Approach starts at 350 mm, decreases by 20 mm to 150 mm, with a 30° half-angle
  cone and ±10 mm axial arrival allowance. A disk is frozen during retries and
  advances only after measured arrival. The disk constrains endpoints, not the
  entire path; a detour may move farther from the object.
- Approaching tries strict pointing first. Only eligible `ik_failed`/`no_path`
  results without a playable strict continuation allow camera-FOV relaxation.
  Keep-distance remains strict. Both passes retain collision, visibility,
  timing, encoder and measured-arrival checks. Finite failure is not proof that
  a target is unreachable.
- Camera defaults: 82 × 68° FOV; identity flange-to-optical mount, explicitly
  assumed and uncalibrated. Point visibility does not establish full-object
  visibility. Added camera geometry is not automatically collision modeled.
- LOST/expired detection pauses while retaining the armed session; fresh
  accepted detections resume from measured joints. Explicit Stop, faults or
  disconnect disarm. Cancelled plans must never execute late.
- Only one controller owns the Pi bridge/serial transport. Disconnect the WebUI
  before using the direct ROS hardware backend.
- **Do not remove the directional-jog block.** The October 6 physical trial
  moved J4 from 39.90° to 141.15° after blocking telemetry delayed Stop.
  A Pi-side independent watchdog and firmware validation are outstanding.
  The detailed incident and guard are in `hardware/README.md`.

## 3. Setup on another Linux machine

Observed development environment: Python 3.12.3, ROS 2 Jazzy, Node 18.19.1.
`environment-observed.txt` captures the installed Python environment, including
CUDA packages; it is evidence, not a portable CPU lockfile. A clean-machine
installation has not been tested during this handover.

Start from the repository root. Install Python 3.12 with venv/development
headers, Node/npm, and a C++17 compiler/CMake. For ROS, install Jazzy, colcon,
rosdep and pybind11-dev using your distribution's ROS setup. Then:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-runtime.txt
npm ci --prefix webui
```

The runtime requirement includes the tested SDK release `pymycobot==4.0.7`.
The original separate SDK checkout was clean at revision
`44d7fb1c34e1d6c3924ec86617e8f9637f449074` of
https://github.com/elephantrobotics/pymycobot.git. If exact source reproduction
is required, install that revision instead:

```bash
.venv/bin/python -m pip install 'git+https://github.com/elephantrobotics/pymycobot.git@44d7fb1c34e1d6c3924ec86617e8f9637f449074'
```

No nested SDK or old catkin/colcon workspace is required for the active URDF;
it is already under `webui/public/models/`. Keep its license and SOURCE.txt.
For ROS dependencies, with rosdep already initialized:

```bash
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths object_tracking_ros2 object_tracking_cpp --ignore-src -r -y
```

The launchers build these two packages into `output/object_follow_ros2/`.
The full detector pipeline is a **separate project**: a sibling `objectScanner`
checkout, its object dataset/model (historical default `my_object_20260925_r2`),
Orbbec camera runtime and calibration must be transferred separately. See
`./start_ros_system.sh --help` for scanner root/dataset/address overrides.
Do not treat that full launcher as a simulation onboarding command.

On the Pi, the bridge depends on the existing `/home/er/Server_280.py`, the
robot's serial/GPIO environment and Atom 6.5 firmware. Follow `hardware/README.md`;
that vendor file and Pi image are not supplied by this repository.

## 4. Hands-on exercise: first day, no physical motion

1. Read sections 1–2 and the incident in `hardware/README.md`.
2. Check model identity without connecting to a robot:

   ```bash
   .venv/bin/python -c 'from webui.hardware_diffusion import verified_status; print(verified_status()["checkpoint_sha256"])'
   .venv/bin/python -c 'from simulation.diffusion.classical_service import info; s=info(); print(s); assert s["status"] == "ready"'
   ```

   The v2 hash must be
   `6582cbf4e1d2575007cfead8f99ed5f15a7b7021083d0eb200c61b224401bddc`.
   If checks fail, restore the pinned files; do not weaken hash/schema checks.
3. Run `./start_v2_simulation.sh`, open http://127.0.0.1:8000, choose a reachable
   fixed target and **Plan learned IK + RRT***. Inspect the route, playback and
   endpoint error. Save a screenshot and the test target/start pose. Try an
   unreachable target and explain the rejection without claiming a proof of
   unreachability. End the server with Ctrl+C.
4. Run the lightweight checks below and save the summary. Read one lifecycle
   test to explain how stale input, LOST and explicit Stop differ.
5. With ROS installed, run
   `./start_object_tracking.sh --dry-run controller_language:=cpp`.
   This starts disarmed without automatic Pi connection. Inspect status in a
   second sourced terminal with `ros2 topic echo /object_follow/status`.
   No measured hardware feedback is expected while disconnected. Ctrl+C exits.
6. Review `test_camera_fallback.py` and `test_full_pipeline.py`. With your mentor,
   run the documented fake-motor suite in the ROS README. The full-pipeline
   test requires the built native executable and uses isolated ROS domain 185.

Done means: both model checks ready, WebUI builds and previews a checked path,
lightweight tests pass, disconnected ROS starts, and you can explain the units,
frames, ownership and pause/stop rules. Keep physical arming for supervised
commissioning after the calibration/Stop work below.

## 5. Verification and evidence

Quick checks (no physical robot required):

```bash
npm run build --prefix webui
OPENBLAS_NUM_THREADS=1 PYTHONPATH="$PWD/object_tracking_ros2:$PWD" \
  .venv/bin/python -m pytest -q object_tracking_ros2/test/test_core.py \
  object_tracking_ros2/test/test_runtime_options.py
```

If pytest is only installed through Ubuntu on the existing development PC,
use system Python for these two dependency-light tests, or install pytest in
the venv as above. For broader ROS, fake-motor and native pipeline tests, use
`object_tracking_ros2/README.md`; those exercise much more than the quick check.

The retained October 20 summary is in
[evidence/ros2-camera-fallback-20261020.json](evidence/ros2-camera-fallback-20261020.json):
141 ROS tests (1 deselected), 1 native pipeline test, 88 shared hardware tests,
47 planner/cancellation tests and 2 focused resume tests passed in their
respective historical runs. Counts may overlap; do not present their sum as a
unique test count. Motors/detections were simulated. Original raw logs and
traces remain in the excluded `output/ros2_camera_fallback_20261020/` archive.
Current handover checks are recorded separately in `HANDOFF.md`.

## 6. Ordered backlog for the new owner

| Priority | Work | Acceptance evidence |
| --- | --- | --- |
| 1 | Reproduce from a clean clone | Install log, model hashes, build, quick checks and disconnected ROS startup |
| 2 | Transfer detector/Pi dependencies | Scanner revision and dataset, camera runtime, Pi bridge/vendor file and firmware inventory |
| 3 | Calibrate wrist camera and table/tool frames | Measured extrinsics/intrinsics, consistent ROS/planner configuration and physical image-bound checks |
| 4 | Qualify Stop and hardware control under supervision | Independent Pi watchdog, disconnect/stale-feedback tests, measured stop latency and travel; keep directional jog blocked until resolved |
| 5 | Qualify the latest approach on hardware | Controlled target tests, measured endpoint error, dynamics, LOST/resume, no late execution after Stop |
| 6 | Broaden planning coverage | Recorded reachable failures, independent geometric/encoder replay, reproducible seeds and budgets |

Do not retrain models merely to start the system. Preserve checkpoint identities,
complete-route checks, rounded-encoder gates and measured-state replanning.
A failed planner run is a useful reproducible fixture, not a reason to bypass
validation. Physical settings must be established with the supervisor before
using `--hardware` or an arming service.

## 7. Troubleshooting and project boundaries

| Symptom | First check |
| --- | --- |
| Model unavailable/hash mismatch | Runtime artifacts, active manifest and URDF; restore exact files |
| ROS import/build failure | Jazzy sourced, Python 3.12 dev headers, pybind11, venv and rosdep dependencies |
| SDK `-1` / missing telemetry | Pi address and TCP 9000, bridge process, exclusive serial ownership |
| Camera target rejected | Optical frame name, timestamp, TRACKING state, depth, measured-feedback bracket |
| `no_path` / `ik_failed` | Save measured start, frozen target/disk, camera settings, seed and budgets |
| Cannot arm HTTP adapter | Matching camera/orientation protocol between frontend and server |

The fixed-target offline arm explainer was last recorded as deployed at
https://arm.gradual-lab.dev/research/mycobot. Its source is locally archived at
`output/anatomy_arm_approach_20261015/`; the hosting website belongs to the
separate `nn_ik/webui` project. Neither hosting state nor remote access was
rechecked for this handover. Preserve the fixed-target replay preference; a
moving-target experiment was intentionally rolled back. Transfer that archive
separately if the junior will maintain the public illustration.

Research training datasets, historical output, local environments and upstream
clones are excluded from the source transfer. See [GITHUB.md](GITHUB.md) for the
publish checklist. No top-level Git repository existed when this handover began.
