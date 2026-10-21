# myCobot 280 Pi: simulation, checked planning and ROS object following

Research and integration workspace for a six-joint myCobot 280 Pi. It includes
an interactive WebUI, learned IK with classical RRT* planning, diffusion motion
experiments, a checked encoder executor, and C++/Python ROS 2 object following.

**Start with the [junior engineer handover](docs/HANDOVER.md).** It contains setup,
a hands-on exercise, the code map, known limitations, and an ordered backlog.
[HANDOFF.md](HANDOFF.md) records the latest session snapshot.

## Current baseline

- Default fixed-target and tracking planner: learned IK + classical RRT*.
- ROS 2 Jazzy: native C++ frontend embedding the shared Python controller;
  Python frontend remains selectable.
- Approaching: 350 → 150 mm in 20 mm stages, strict pointing first, then a
  guarded camera-FOV fallback when the strict search has no playable continuation.
- Historical October 20 validation passed with fake motors and synthetic
  detections. Physical calibration, dynamics and stopping behavior remain open.
- Atom 6.5 directional jog is blocked following a physical overtravel incident.
  Preserve that guard; see [hardware notes](hardware/README.md).

## First launch

After completing the dependency setup in the handover:

```bash
npm ci --prefix webui
./start_v2_simulation.sh
```

Open http://127.0.0.1:8000. This entry point explicitly disables hardware commands.
Use it for onboarding. The full ROS launcher can connect to hardware and start
an external detector; follow the handover before using it.

## Documentation

| Area | Entry point |
| --- | --- |
| Transfer, setup, hands-on exercise, next work | [Handover](docs/HANDOVER.md) |
| GitHub inclusion and transfer checks | [GitHub preparation](docs/GITHUB.md) |
| WebUI and planner selection | [WebUI](webui/README.md) |
| ROS interfaces, frames, lifecycle, camera policy | [ROS controller](object_tracking_ros2/README.md) |
| Serial bridge and hardware incident | [Hardware](hardware/README.md) |
| Original simulation experiments | [Simulation](simulation/README.md) |
| Diffusion, measured-feedback integration and training | [Diffusion](simulation/diffusion/README.md) |
| Latest retained historical test summary | [October 20 evidence](docs/evidence/ros2-camera-fallback-20261020.json) |

Older component documents describe successive experiments. Prefer the current
source and the handover for the active baseline. Output directories referenced
by older documents are local research archives and are excluded from Git.
