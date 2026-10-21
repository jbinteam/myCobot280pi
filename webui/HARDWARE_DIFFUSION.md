# v2 hardware trials

For continuous mouse tracking, follow [Hardware mouse aim](HARDWARE_MOUSE_AIM.md).
The steps below cover a fixed target and its path waypoint executor.
Open **Advanced controls** to access the fixed-target buttons and target fields.
The frame, table and movement confirmations are grouped in **Prepare the arm**.

The pinned `diffusion_motion_v2` model can prepare a reviewed path from fresh,
stopped robot telemetry. It still uses collision-free IK to obtain its goal joint
conditioning. Cartesian v3/v4 checkpoints are rejected by this hardware adapter.
Planning sends no position, power or refresh-mode command.

Start the Pi bridge described in [hardware/README.md](../hardware/README.md),
then start the PC UI from the project root:

```bash
./start_v2_hardware.sh
```

The launcher verifies the v2 checkpoint/schema, builds the UI, and enables hardware
controls at http://127.0.0.1:8000. Startup does not connect or move the robot. The
existing `./start_webui.sh --hardware` entry point also exposes these controls.

1. Connect using the Pi's current IP and port 9000. Confirm that displayed live
   joints match the arm. Select **Diffusion v2 · recommended**.
2. Use **Use live flange pose**, then change the target by only a few millimeters.
   Targets use `g_base` coordinates at `joint6_flange`; the directed flange Z axis
   is constrained, and spin around Z is free. Local target offsets are resolved
   using the fresh live flange rotation on the server.
3. Press **Plan from live joints**. Play/scrub the entire predicted path and
   inspect the joint motion. A failed or unsafe rollout produces no execution plan.
4. Verify the URDF joint zeros/axes and flange frame. Check the frame confirmation.
   Verify that the actual table matches the displayed table at **−35 mm in
   `g_base`**, then check the scene confirmation. This is an operator verification;
   the software has no sensor to establish calibration. If the table differs,
   do not execute this scene. Added tools, cables and external obstacles are not
   modelled; the first-trial scene uses **no block**.
5. Clear the entire path, check **Enable movement — path is clear**, and retain
   **Arm speed 5** for the first trial (allowed 1–80% for fixed targets).
6. Press **Execute planned path on robot**. Keep the browser open. Observe actual
   joint telemetry and the estimated flange error, then stop before changing targets.

Plans expire after 60 seconds, execute once, and are invalidated by changed
connection/command generation or more than 1° start-pose drift. Input edits discard
the browser's plan. Execution accepts a server-owned plan ID and confirmation flags;
the browser cannot submit joint paths. The checkpoint and scene are checked again
before execution.

Checked quintic path corners are preserved and subdivided into **at most 2° per
joint** steps. The complete original preview and hardware waypoint path receive
independent self/table/limit checks with 0.01 rad segment sampling and 1 mm model
clearance. During execution, fresh telemetry and each live-to-command segment are
checked in a new instance of the same pinned collision scene. No unsafe replacement,
collision bypass, or automatic command retry is used.

The executor sends each target once with the SDK asynchronous encoder. Interior
points on a checked straight leg allow **≤3° per-joint tracking error** and may
advance while the arm is moving. Every retained corner and the final endpoint
still require **stopped motion and ≤1.5° joint error**, matching continuous v2
hardware execution's allowance for servo deadband. The corner is commanded and
settled before starting the next leg. A slowly changing joint can remain still
across several path steps before its cumulative command crosses the deadband.
This does not relax final Cartesian acceptance. Live-to-command change is capped
at **4° per joint**, including tracking slack; lag cannot queue unbounded commands.
Each actual live-to-command segment is checked independently for collision.
Execution verifies firmware refresh mode 1 so interior commands replace the
active target rather than queue; the previous mode is restored on completion,
Stop, failure or disconnect when the transport remains available.
The executor refuses plans over 200 waypoints, checks
power/servo/pause/error state, and retains the existing 10-second waypoint,
180-second execution, and 5-second browser-heartbeat timeouts. Stop, Escape,
Disconnect, or loss of the browser lease cancels remaining commands.

Fixed-target previews use the selected SDK percentage to calculate their joint
velocity allowance and pace waypoint sends. Telemetry, command audits and
settling can extend execution. The firmware controls interpolation between
targets; the preview does not guarantee measured physical speed/acceleration.
Completion requires estimated URDF FK error ≤5 mm and pointing error ≤3°; neither
is independent Cartesian measurement. Software Stop depends on the PC/network/Pi.
Physical v2 accuracy and smoothness still require the supervised trial above.
Hardware mouse aim uses the continuous v2 executor when v2 is selected; Classical
IK retains its existing mouse controller. The two execution modes have different
reference timing and limits.

Implementation: `hardware_diffusion.py`, `HardwareIK.prepare_diffusion`, and
`POST /api/ik/diffusion/plan`; execution uses the existing `/api/ik/execute` and
`/api/ik/heartbeat` endpoints. Simulation diffusion endpoints keep their hardware
gates. Fake-arm Python/browser tests send no physical motor commands.

The fixed-v2 velocity-profile mode allows arm speed 1–80%. Analytic joint velocities use a nominal per-joint allowance of up to 120°/s at 80%; acceleration, flange and communication tracking limits can reduce actual speed or stop execution. Legacy bounded encoder trials remain limited to 5%. Changing speed requires planning and reviewing again.
