> The web UI now defaults to five-constraint XYZ + flange-Z-direction control.
> See [POSE5.md](POSE5.md) and [SEED5.md](SEED5.md) for the separately trained 5D Seed NN and PRM workflow. This document describes the retained XYZ-only experiments.

A [diffusion motion-planning experiment](diffusion/README.md) now uses
this arm's URDF in PyBullet, RRT* expert demonstrations, a conditioned temporal
U-Net and checked receding-horizon reference execution. Its trained checkpoint,
1,000-episode dataset and held-out evaluation are separate from WebUI Seed5.

Reusable joint-space PRM planning is available through Python, CLI and the dry-run
`/api/sim/prm` endpoint. See [PRM.md](PRM.md) for the inspected NN interfaces,
build/query/benchmark commands, endpoint/path diagnostics and validation limits.
PRM now checks conservative URDF self-collision geometry by default. See
[SELF_COLLISION.md](SELF_COLLISION.md) for installation, cache profiles and limits.
Simulation [mouse aim](MOUSE_AIM.md) first tries collision-checked local 5D IK
from the current joints to aim at a camera-plane cursor with a 50 mm standoff
and free local tool-Z spin. If the complete local path or goal is rejected,
it uses the trained 5D NN, numerical and self-checked direct/PRM fallback.
That fallback includes [bounded solver and seed-coverage recovery](IK_RECOVERY.md),
with one expansion to eight IK goals when the initial routing choices fail.
[Guided joint-space repair](GUIDED_REPAIR.md) can add temporary samples after
a roadmap routing failure. The UI enables it for PRM previews and mouse aim
through the **Repair blocked routes** checkbox.

# Phase 1: myCobot IK simulation

This ports seven controller ideas from `~/codex/projects/nn_ik` to the actual
six-joint myCobot 280 Pi URDF. It adds a matched classical null-space baseline.
The original project studies a planar 3R arm; its trained weights and analytical
planar joint-limit detours cannot be reused for this arm.

## Run and inspect

```bash
./start_webui.sh
```

Open http://127.0.0.1:8000 and select the XYZ-only task to use these eight methods. The simulation-only server automatically compares a reachable example in the selected task.
Enter target XYZ in millimeters, edit the starting joint pose, or select one of
24 held-out cases. Run one method or compare all eight. Select a table row or
chart legend entry, then Play/Pause or scrub through the rollout. The orange
marker is the target; the colored line is the simulated tool path. Playback
speed changes animation only. Editing joints or choosing a case resets the
experiment start; rerunning without editing preserves the previous start for
fair comparisons.

The server defaults to simulation only and rejects hardware connect, move and
jog requests. Simulation code imports no SDK or robot transport. It never
sends computed poses to hardware. The physical control UI is hidden.

## Adapted methods

| Method | Implementation |
|---|---|
| Pseudoinverse | 23/24 | 10.40 |
| DLS | 24/24 | 0.92 |
| Direct NN | 0/24 | 37.31 |
| Conditional NN | 0/24 | 155.07 |
| Differential NN | 0/24 | 177.68 |
| Classical null-space | 24/24 | 0.93 |
| Neural null-space | 24/24 | 0.93 |
| NN seed + DLS | 24/24 | 0.92 |

Source ideas: `nn_ik/rrr_neural_ik/architectures/`, `classical_ik/`, and its
experiment, hybrid null-space and pose-conditioned seed documents.

FK and the analytical position Jacobian come directly from the same local URDF
rendered in the browser. The endpoint is `joint6_flange`; no gripper/tool offset
is included. Coordinates are in the URDF base frame, meters internally, degrees
at the UI boundary. Only **XYZ position** is controlled; orientation is free.
Joint bounds intersect URDF limits with the existing SDK limits. No dynamics,
collision detection, Cartesian trajectory planning or hardware calibration is
included. Joint clipping can stall a solver at a limit; the planar analytical
detour strategy has not been ported.

Every method has a 0.05 s integration step and the same configurable joint
velocity vector norm cap (default 1.2 rad/s), 240-step budget, and 1 mm success
tolerance. This simulated speed is independent of the SDK speed setting (30).
DLS uses a position scale of 0.3 m, gain 3 and damping 0.04. Null-space policies
use gain 0.3; projection is exact, although finite steps and limit clipping can
still change the primary task. Seed selection minimizes normalized FK error
squared plus 0.003 times joint travel squared. Seed motion is speed capped and
counts toward the same budget as refinement; it never teleports.

## Verified joint limits

Read-only firmware queries on 2026-09-30 confirmed the connected Pi's Atom 6.5
limits: J1 ±168°, J2 ±135°, J3 ±150°, J4 ±145°, J5 ±165°, J6 ±180°.
Raw readings, current angles and the still-reported error 4 are recorded in
`artifacts/robot_limit_survey.json`. No physical limit-seeking motion was used.
These values match the [older Pi manual](https://docs.elephantrobotics.com/docs/mycobot-pi-en/2-serialproduct/2.1-280/2.1.2.1%20Introduction%20of%20product%20parameters.html).
The SDK's newer J2 ±140° / J4 ±150° ranges were too wide for this firmware.

`joint_limits.json` records the verified profile. All solvers, training and web
pose commands use the intersection of firmware, SDK and URDF bounds, inset by
2° and rounded inward to 0.1°. The operating ranges are J1 ±165.9°, J2 ±133°,
J3 ±147.9°, J4 ±143°, J5 -152.9° to +157.9°, J6 ±177.9°. The CLI's individual
joint commands also reject targets beyond these ranges. Joint jogging is a
velocity command; the firmware remains responsible for its hardware limits.

All five XYZ networks were retrained on CUDA with the corrected operating
ranges (24,000 training / 2,048 validation samples). The model loader rejects
weights with mismatched limits, since joint normalization depends on them.
The previous weights and benchmarks are in `artifacts/backups/`.

## Trained artifacts and measured results

Five new MLPs use two 128-unit tanh hidden layers. Training uses 24,000 FK-derived
samples (seed 42), with a separate 2,048-sample validation set (seed 43). Three
quarters of targets are nearby joint perturbations; the rest are global poses.
Direct and seed networks use FK losses, the conditional/differential networks
imitate DLS, and the null-space network learns the same centering posture policy
as its classical baseline. Validation loss selects checkpoints. `train.py`
exports PyTorch checkpoints and portable NumPy weights; serving requires only
NumPy, not PyTorch or a GPU.

Saved benchmark: 24 independent targets, seed 9001, 12 nearby and 12 global,
with every fourth start at the zero pose. Targets are reachable by construction. Targets were regenerated within the new
limits, so these results are not a like-for-like comparison with the old table.
One training seed and this small case set are exploratory evidence, not a
reliability or superiority claim.

| Method | Reached within 1 mm | Mean final error (mm) |
|---|---:|---:|
| Pseudoinverse | 23/24 | 10.40 |
| DLS | 24/24 | 0.92 |
| Direct NN | 0/24 | 37.31 |
| Conditional NN | 0/24 | 155.07 |
| Differential NN | 0/24 | 177.68 |
| Classical null-space | 24/24 | 0.93 |
| Neural null-space | 24/24 | 0.93 |
| NN seed + DLS | 24/24 | 0.92 |

The standalone neural controllers are not reliable at this training budget.
Their failures remain visible for diagnosis. One-step imitation loss does not
establish stable closed-loop control. Neural null-space currently imitates the
classical centering policy; it is not a learned collision or manipulability
objective. Future training can address rollout stability, branch diversity,
singularities and limit avoidance before any hardware phase.

`artifacts/benchmark.json` contains individual case outcomes, travel and CPU
rollout timings. `training.json` records losses/configuration and the URDF SHA256;
`models.json` contains the inference weights. The loader rejects a changed URDF
fingerprint or joint-limit normalization. Training used the PC's CUDA GPU; timing excludes network transfer
and is distinct from simulated elapsed time.

## Reproduce

Use a Python environment with NumPy and PyTorch for training (system `python3`
has both on this PC). The project `.venv` has NumPy for serving/benchmarking.

```bash
python3 -m simulation.train --device auto --steps 1200
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.solvers
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s simulation/tests -v
```

Restart the server after replacing model weights. Then reload the page to see
the new benchmark and training metadata. Browser integration tests require the
simulation backend at port 8000 and test zero physical POST requests, playback,
FK parity with the URDF renderer, failure reporting and mobile layout:

```bash
cd webui
npx playwright test
```
