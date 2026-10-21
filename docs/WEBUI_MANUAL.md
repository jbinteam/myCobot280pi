# Running the myCobot WebUI

This guide covers the local browser interface on the development PC. Start with
simulation; it needs neither a Pi nor a connected robot. ROS and the separate
objectScanner project are not required for this workflow.

Run all PC commands below from the repository root. The normal browser address
is **http://127.0.0.1:8000** on the same PC that runs the server.

## 1. First-time setup

The observed development environment uses Python 3.12.3 and Node.js 18.19.1.
Install Python 3.12 with venv support, Node.js/npm, and use a WebGL-capable browser.
The dependency files describe a CPU-capable runtime; CUDA is optional. A fresh
machine installation has not yet been qualified.

```bash
git clone https://github.com/jbinteam/myCobot280pi.git
cd myCobot280pi
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-runtime.txt
npm ci --prefix webui
```

If the repository and `.venv` already exist, use that checkout and environment.
No environment activation is needed: launchers call `.venv/bin/python` directly.
After pulling dependency changes, rerun the pip install and `npm ci` commands.

Check the retained model/geometry files and planner readiness:

```bash
.venv/bin/python scripts/verify_handover_artifacts.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -c 'from webui.hardware_diffusion import verified_status; print(verified_status()["checkpoint_sha256"])'
OPENBLAS_NUM_THREADS=1 .venv/bin/python -c 'from simulation.diffusion.classical_service import info; s=info(); print(s); assert s["status"] == "ready"'
```

The pinned diffusion v2 hash is
`6582cbf4e1d2575007cfead8f99ed5f15a7b7021083d0eb200c61b224401bddc`.
The v2 launchers check this model even when you intend to use learned IK + RRT*.
Restore missing or mismatched artifacts before continuing; retain the integrity
checks. See [handover setup](HANDOVER.md#3-setup-on-another-linux-machine) for
environment details.

## 2. Everyday startup

```bash
./start_v2_simulation.sh
```

Wait for the model check, Vite build, and Python server startup, then open
http://127.0.0.1:8000. Keep the terminal running. Each launch rebuilds the frontend;
you do not need a separate Vite process for normal use.

The page should show **Simulation only**. This launcher locks out hardware
commands, including attempts to enable them through the Mode selector.

| Launcher | Startup behavior |
| --- | --- |
| `./start_v2_simulation.sh` | Checks pinned v2; locks the server to simulation. Recommended first run. |
| `./start_v2.sh` | Checks pinned v2; starts in simulation and allows mode changes in the UI. |
| `./start_v2_hardware.sh` | Checks pinned v2; starts with hardware controls enabled. Does not connect or move the arm. |
| `./start_webui.sh --simulation-only` | Builds and starts a simulation-locked server without the launch-time v2 precheck. Planner readiness checks still apply. |

With `start_v2.sh`, **Mode** switches between **Simulation only** and
**Simulation + hardware**. The mode is shared across tabs. Returning to simulation
stops tracking, disconnects the arm, discards hardware plans, and reloads the
workspace with confirmations cleared.

## 3. Try the simulation

### Fixed-object approach demo

1. In **Approaching simulation**, uncheck **New random object after completion**
   for a single round. It is enabled by default and otherwise repeats.
2. Click **Start approach demo**. The demo loads a starting pose and a fixed orange
   object, then approaches in 20 mm stages toward 150 mm. The first distance comes
   from the starting pose; a start closer than 150 mm holds its initial distance.
3. Watch the checked blue route and distance/status readouts. Enable **Show RRT*
   search** to see the planner's exploration; those branches are not motion commands.
4. Use **Pause**, **Resume**, and **Stop approach**, or Escape to stop.

The default camera-FOV fallback tries strict pointing first, then relaxes
orientation only for eligible failures without playable motion. A failed or
partial stage does not count as arrival. The wrist-camera mount/FOV are assumed,
not physically calibrated.

### Mouse tracking

1. Click **Load visible tracking demo** for a starting pose with a visible target.
2. Retain **Learned IK + classical RRT***, the default when its model is ready.
3. Click **Start mouse aim**, enter the scene, and move gently.
4. Read the status if the preview holds: visibility, collision, or planning checks
   can reject a target. Press Escape or **Stop mouse aim** to stop.

Drag to orbit, Shift + drag or right drag to pan, and scroll to zoom. The orbiting
viewport is separate from the wrist camera used by visibility checks.

### Fixed-target planning

1. Open **Advanced controls** and choose a target/start pose in the simulation controls.
2. Check the target fields and units: positions/offsets are in millimeters;
   displayed joint angles are in degrees. The default task constrains XYZ and
   flange Z direction, leaving spin about flange Z free.
3. Click **Plan learned IK + RRT***. Diffusion is available as an alternative.
4. After a valid result, use **Play** and inspect the full route and endpoint error.
   A planning failure produces no accepted motion; it does not prove the target
   is unreachable.

## 4. Hardware entry point

Use hardware only as a supervised commissioning workflow. Physical stopping,
camera calibration, and dynamics remain unfinished acceptance work. The software
Stop depends on the PC, network, and Pi; it is not a hardware emergency stop.
Directional jog on Atom 6.5 remains blocked after an overtravel incident.

1. Follow [Pi bridge installation](../hardware/README.md). The Pi needs its existing
   vendor `/home/er/Server_280.py` and the repository's `pi_server.py` and
   `pi_transport.py` installed in `/home/er/mouse_control_server/`.
2. On the Pi, after stopping the previous server, run the following with the Pi's
   actual bind address (the address shown is the lab default):

   ```bash
   cd /home/er
   python3 -u mouse_control_server/pi_server.py 192.168.77.225
   ```

   Only one process may own serial `/dev/ttyAMA0` and TCP port 9000. Do not run
   direct ROS hardware control concurrently with WebUI control.
3. On the PC, run `./start_v2_hardware.sh` and open http://127.0.0.1:8000.
4. Enter the current Pi IP and port **9000**, then **Connect**. Verify live joint
   telemetry. Connection alone does not power up servos or command movement.
5. For a fixed target, open **Advanced controls**, retain learned IK + RRT*, use
   **Use live flange pose**, make a small target adjustment, and set **Arm speed 5**
   before **Plan from live joints**. Inspect the whole preview.
6. Verify the actual flange frame, joint zeros, and table against the modeled
   scene. Complete the frame/table and movement confirmations in **Prepare the
   arm** only after checking them. Then use **Execute planned path on robot**
   for the reviewed plan. Keep the browser open and monitor actual telemetry.

Changing speed or target requires replanning and review. For executor details
and constraints, read [fixed-target hardware](../webui/HARDWARE_DIFFUSION.md)
and [hardware mouse aim](../webui/HARDWARE_MOUSE_AIM.md) before execution.

## 5. Stop and restart

- Simulation: stop the active demo/tracking with its Stop button or Escape.
- Hardware: click **Stop robot**, confirm the arm has stopped, then **Disconnect**.
- In the server terminal, press **Ctrl+C**. For development mode, stop both terminals.
- Restart with the chosen launcher and reload the page to obtain a fresh session.

Closing a browser tab is not the normal shutdown procedure. If communication is
lost, use the lab's physical stop procedure; a network Stop may not arrive.

## 6. Frontend development

Run these in separate terminals at the repository root:

```bash
# Terminal 1: API server, locked to simulation
OPENBLAS_NUM_THREADS=1 .venv/bin/python webui/server.py --simulation-only
```

```bash
# Terminal 2: frontend with live reload
npm run dev --prefix webui
```

Open **http://127.0.0.1:5173**. Vite forwards `/api` to port 8000. Both services
bind to loopback. Use the documented ports: request-origin checks also expect
8000/5173. `npm run preview` alone does not supply the robot API.

## 7. Troubleshooting

| Symptom | Action |
| --- | --- |
| `.venv/bin/python` missing or Python import failure | Run the environment/dependency setup from the repository root. Ensure packages were installed with `.venv/bin/python -m pip`. |
| `vite: not found` or frontend dependency failure | Run `npm ci --prefix webui`, then restart the launcher. |
| Model precheck fails or planner is unavailable | Run the artifact and readiness checks above. Restore the pinned files; inspect the terminal error. |
| Browser cannot connect | Wait for the build to finish; inspect terminal errors. Open the URL on the server PC. The server does not bind to the LAN/Pi address. |
| Address already in use | Inspect `ss -ltnp 'sport = :8000'` (or `:5173` for Vite). Stop the identified old server in its terminal before restarting. |
| Invalid local control session / HTTP 403 | Reload the page after restarting the server; use the documented localhost URL. |
| Hardware mode unavailable | A simulation-locked launch deliberately disables it. Stop the server and use `./start_v2.sh` when hardware work is intended. |
| Pi connect fails or joint telemetry reports SDK `-1` | Check the current Pi IP, network, TCP 9000 listener, and exclusive bridge ownership. Restore the connection and explicitly reconnect. |
| Preview holds or planning fails | Inspect status for collision, visibility, invalid start, or exhausted search. Try the visible demo or another target; retain validation checks. |
| Old frontend after editing | Normal mode needs a rebuild/restart and page reload. Use development mode for live reload. |

For a read-only HTTP check while the server is running:

```bash
curl --fail --silent http://127.0.0.1:8000/api/state | .venv/bin/python -m json.tool
```

In a simulation-locked session, expect `hardware_enabled: false`,
`mode_selectable: false`, and `control_mode: "simulation"`. Do not share this
response: it also includes the local session token.

See the [component reference](../webui/README.md) for planner internals and the
[project handover](HANDOVER.md) for the broader ROS workflow and remaining work.
