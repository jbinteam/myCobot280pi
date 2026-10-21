# Phase 2: real-arm IK execution

The web UI plans XYZ or five-constraint pose IK from live myCobot joint telemetry and executes the
reviewed joint-space path through the existing TCP controller. Eight methods
are available in XYZ mode; the five-constraint task uses three classical solvers. Only converged results may execute. Start with DLS in five-constraint mode. NN seed + DLS supports XYZ-only mode. Standalone neural models remain exploratory and often fail to converge.

## Start

On the Pi, run the existing `Server_280.py` on TCP port 9000. On the PC:

```bash
./start_webui.sh --hardware
```

Open http://127.0.0.1:8000. The flag exposes hardware controls; startup does not
connect or move. Without the flag, the server still rejects every hardware
connect/move/jog/IK endpoint and remains phase 1 simulation only.

## First motion

1. Click **Connect** using the Pi address (`192.168.77.225:9000` by default).
   Check live angles against the arm. Connect and planning send no motion.
2. Click **Use live flange pose**, then change one coordinate by a few mm.
   This also copies the current flange tilt. In 5D mode you may change Orientation
   X/Y while local flange-Z spin stays free. See [orientation convention](../simulation/POSE5.md).
   Target coordinates are **URDF base coordinates**, in mm, at `joint6_flange`.
   They are not assumed to equal SDK `get_coords()` or world/workbench coordinates.
3. Select **DLS** and click **Plan from live joints**.
   The server reads the current stopped pose and solves from it. Editor poses
   and held-out cases cannot supply the start of a hardware plan.
4. Play/scrub the predicted path and inspect the joints. The model becomes a
   preview while planning. In XYZ-only mode orientation is free; joint 6 and wrist
   posture may change. In 5D mode the directed
   flange Z axis is constrained and spin around that axis is free.
5. Verify the joint zero positions, joint direction/axes and base/flange frame
   match the installed arm. Check the corresponding checkbox. This checkbox
   records your verification; the software has no automatic calibration or
   external sensor to establish it. For an attached tool, the current controlled
   point is still the flange, not the tool tip. Account for the tool's clearance.
6. Clear the full path and check **Enable movement — path is clear** in the
   joint panel. Set **IK SDK speed** (default 30, allowed 1–80%).
7. Click **Execute planned path on robot**. The model follows live telemetry,
   the progress count advances, and a separate error chart shows wall time.
8. **Stop robot**, Escape, disabling movement or Disconnect cancels the queue.
   Keep the browser session open during execution; a lost heartbeat stops it.

Targets and solver settings are locked during hardware motion; Stop before editing
a new goal. A plan expires after 60 seconds and can execute once. Input edits invalidate
the browser's plan. The server rechecks live angles and refuses a plan after
intervening commands, a changed connection, or >1° change in any starting joint.
A rejected/non-converged plan is never replaced with a silent fallback.

## Execution contract

- The server owns the trajectory. Execute accepts only its plan ID, speed,
  movement-enabled flag and frame-verification flag, never browser joint arrays.
- Solver frames are compressed and split into waypoints with at most 3.5° change
  per joint. Actual telemetry must be within that bound plus 0.5° arrival slack
  before issuing the next waypoint.
- Each command uses `send_angles(..., _async=True)`, matching the old Atom
  firmware's lack of motion ACKs. The next waypoint waits for valid telemetry,
  motion status stopped, and joint errors ≤0.5°. Commands are not streamed at
  the simulated 0.05 s tick. Actual servo interpolation and dynamics can differ
  from the preview.
- Power, servo enable, pause and robot error flags are checked before execution
  and each command. Overlapping move/jog/IK requests are refused.
- One waypoint has a 10 s settling timeout; the whole run has a 180 s limit.
  The first motor command waits for the owning browser to receive the execution
  response and send a heartbeat. Subsequent heartbeats arrive every second;
  absence for 5 s cancels execution and
  attempts Stop. Server execution continues between browser polls, but cannot
  continue indefinitely without the owning browser session.
- Read/send failure, tracking drift, unexpected motion, timeout, Stop and
  disconnect cancel all remaining waypoints. No automatic retry or replay occurs.
- Completion requires the final joint tolerances and ≤5 mm **estimated** flange
  position error, plus ≤3° pointing error for five-constraint plans. Both the live XYZ and error plot are URDF FK computed from
  joint telemetry, not independently measured Cartesian accuracy. Simulator
  tolerance (default 1 mm) is separate from this execution check.

The manual controller-IK path above does not include collision checks.
[Hardware mouse aim](HARDWARE_MOUSE_AIM.md) uses the separate self-checked
NN/IK/PRM path and validates live-to-command segments. Neither path includes
external-obstacle checks, dynamics simulation, force control, gripper
control, tool offset calibration, full six-constraint orientation control or external position
feedback. The software Stop uses the PC/network/Pi and is not a hardware
emergency stop. Checking the path and frame is necessary before a real move.

## Files and verification

- `hardware_ik.py`: live-start planning, one-use plans, waypoint worker/lease.
- `server.py`: persistent transport, serialized SDK access, existing Stop and
  health checks, hardware mode gating, `/api/ik/{plan,execute,heartbeat}`.
- `src/hardware-ik-controls.js`: planning/execution UI and live error plot.
- `src/simulation-controls.js`: shared URDF preview and solver plots.

Tests use fake hardware and mocked browser endpoints; they do not physically
move the arm. They cover read-only planning, bounded waypoints, one-use plans,
expiry, pose drift, non-convergence, required flags, Stop, heartbeat loss,
stalling, unexpected motion, failed sends, disconnect and overlapping commands.
The installed robot's calibration and physical performance remain to be tested
with the reviewed first-motion procedure above.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s webui/tests_python -v
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s simulation/tests -v
cd webui
npx --no-install playwright test
npm run build
```

Run the browser suite against a phase 1 backend (without `--hardware`); phase 2
browser tests mock the hardware API explicitly.
