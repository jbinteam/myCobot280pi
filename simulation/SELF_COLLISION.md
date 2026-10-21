# Self-collision upgrade for PRM planning

PRM builds, `Roadmap()` queries and `/api/sim/prm` now default to **joint limits
plus conservative self-collision validation**. NN inputs and checkpoints are
unchanged. The existing solver comparison, workspace estimator and hardware
execution interfaces remain separate; this upgrade is used by PRM dry runs.
No physical robot motion was performed.

The WebUI **Plan self-checked PRM** button uses this dry-run endpoint and the
existing Play/scrub preview. It displays joint/FK waypoints, goal-candidate
attempts and failure diagnostics. Starting a PRM preview invalidates any older
hardware execution plan in the browser. See [WebUI setup](../webui/README.md).

## Geometry and checker

`self_collision.py` loads every URDF `<collision>` mesh for all eight links and
builds a solid convex hull for each collision element. It checks 21 link pairs;
the seven directly connected URDF parent/child pairs are excluded because those
parts form mechanical joints. Two-hop links, the base and the flange are checked.
No additional pairs are silently excluded based on sampled poses.

The collision geometry uses `ARM.link_transforms(q)` from the same URDF chain as
FK. COLLADA scene transforms, asset units, URDF mesh scale and collision origin
are applied. Some meshes are in meters, others in millimeters. The stale wrist
material reference is repaired in memory as in the viewer. Missing, empty or
unsupported collision geometry fails closed. Original model files are untouched.

FCL performs convex solid intersection and distance queries, including fully
contained objects. Axis-aligned bounding boxes only reject clearly separated
pairs before the narrow-phase test. Default minimum clearance is **1 mm**;
`--self-clearance-mm` / `self_clearance_mm` can select 0-10 mm. Native collision
object transforms are mutable, so pose checks are serialized for thread safety.

Convex hulls enclose the original mesh: they fill concavities and can reject
configurations that would be free under exact mesh geometry. This is a deliberate
conservative proxy, not a calibrated geometric model of the installed arm. See
[python-fcl](https://github.com/BerkeleyAutomation/python-fcl) and
[Trimesh collision documentation](https://trimesh.org/trimesh.collision.html).

## Integration and persistence

The checker runs on sampled/reused roadmap nodes, proposed IK endpoints, local NN
rollout transitions, direct connections, graph edges, shortcuts and final preview
segments. A numerically accurate but self-colliding goal is rejected. Numerical
refinement is an endpoint computation, not executed motion: its intermediate
iterations need not be collision-free. Actual planned motion is checked separately.
An in-limit NN seed may therefore be numerically refined even if its starting
configuration is self-colliding; the resulting goal still must pass FK and
self-collision checks. The seed itself is never sent as a motion waypoint.

For 5D queries the WebUI passes the selected classical controller as
`pose5_method`. It proposes/refines numerical endpoints, with bounded DLS tried
from the same seed if the selected method's endpoint is rejected. Standalone
API/CLI defaults remain bounded DLS; the CLI exposes `--pose5-method`. Attempts
report the actual refinement method and both XYZ and direction errors. The
separate 5D Seed NN is documented in [SEED5.md](SEED5.md). Its four raw guesses
are checked before independent refinement and numerical fallback. This improves endpoint coverage without changing
path validity rules or re-solving known waypoints.

Results identify `joint_limits_and_self_collision`. Invalid starts and rejected
endpoint attempts include collision/clearance link-pair diagnostics. Limits,
IK failure, path failure and local rollout failure remain distinct. Failure within
a search budget does not prove physical unreachability.

Cache assumptions identify convex representation, pair exclusions, clearance,
checker version, library versions and mesh units/bounds, in addition to the
existing robot/mesh hashes, frames, units and limits. Changing these assumptions
rejects a saved roadmap. Mesh changes invalidate the cached checker; URDF changes
require restarting the process to reload FK. Old limit-only roadmaps cannot be
silently loaded into self-collision queries.

`roadmap.json` is rebuilt for self-collision. The prior graph is retained as
`roadmap_limits.json`. The HTTP service validates a persisted graph once and
reuses it; a changed file or geometry/validity profile invalidates that in-memory
cache. Every returned path is still rechecked. Initial load and building are
slower because they validate all edges; warmed queries reuse that work.

## Install and reproduce

Use the project Python 3.12 environment:

```bash
.venv/bin/python -m pip install -r simulation/requirements-collision.txt

# Default self-collision profile, 1 mm clearance and 0.04 rad per-joint sampling.
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.prm \
  --samples 400 --neighbors 12 --max-step-rad .04 --seed 9001 \
  --dataset simulation/artifacts/dataset.npz

OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.planning \
  --start-deg 0 -40 -130 80 0 50 --target-mm 100 0 200

OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.benchmark_prm \
  --count 24 --samples 400 --seed 9001 \
  --output simulation/artifacts/benchmark_prm_self.json

OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s simulation/tests -v
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s webui/tests_python -v
```

For an explicit limit-only ablation, pass `--collision limits` to build/query/
benchmark. Use a separate cache/output file. A low-level `JointSpace()` still
means the limit-only configuration box; `planning_space()` and `Roadmap()` default
to self checking. Generic callbacks remain supported.

The HTTP request accepts `collision` (default `self`) and `self_clearance_mm`
(default 1) in addition to existing query fields. Request and saved-cache profiles
must match. Explicitly requesting limits cannot use a self-collision cache or
mislabel its results.

## Reproducible self-collision path

`fixtures/self_collision_path.json` fixes valid start and goal configurations.
The straight segment midpoint intersects the convex proxies for
`g_base/joint6` and `joint1/joint6`. A centered-configuration detour is valid;
the default 400-node roadmap finds a shorter validated route. This isolates path
planning from IK accuracy using a known goal configuration:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python - <<'PY'
import json
from pathlib import Path
import numpy as np
from simulation.prm import Roadmap
f = json.loads(Path('simulation/fixtures/self_collision_path.json').read_text())
start, goal = np.deg2rad([f['start_deg'], f['goal_deg']])
road = Roadmap.load('simulation/artifacts/roadmap.json')
print('Direct valid:', road.space.edge_valid(start, goal, road.config.max_step_rad))
print(road.plan(start, [goal], shortcut=True)['reason'])
PY
```

`tests/test_self_collision.py` checks FK/link frames, all loaded geometry and units,
pair exclusions, known clear/colliding states, solid containment, accurate but
colliding goals, intermediate collision rejection, a genuine URDF-proxy detour,
local rollout rejection, clearance/profile invalidation and persisted graph reuse.

The new benchmark keeps the same 24 original start/target cases and separately
counts invalid starts under self-collision rules. These cases were originally
sampled using joint limits only, so overall success rates cannot be compared
blindly against earlier limit-only reports. Direct/PRM ablations share goal
candidates; the fixed collision-path fixture measures routing independently.
The original `benchmark_prm.json` and `benchmark_prm_pose5.json` remain historical
limit-only results.

## Remaining limits

Self-collision checking uses conservative convex proxies and discretized motion
samples, with default maximum change of 0.04 rad per joint. Thin collision regions
can fall between samples. This is not continuous articulated collision detection.
Smaller steps and conservative clearance reduce, but do not eliminate, that risk.
External obstacles, the table, added tools/grippers, cables, calibration errors,
dynamics and tracking deviations are not checked. Directly connected pairs are
excluded and their physical mechanical constraints remain an assumption. A
successful dry-run is not a physical safety certification or an executable
hardware plan.

## Recorded validation and results

The simulation suite now has **64 passing tests**, including regressions for
colliding optimizer seeds, selected 5D solvers and validated endpoint fallback.
The previously run fake-hardware/API suite passed **42 tests**. Recorded benchmark
numbers below precede these candidate-generation improvements; their 5D default
uses bounded DLS.
The default roadmap retains 400 valid nodes and 3,003 validated edges; 200 nodes
reuse FK training configurations. Building took about 28.8 s on this machine,
with geometry preparation and dataset loading outside that growth measurement.
Cold cache loading rechecks persisted edges; warmed HTTP queries reuse the graph.

On the original 24 XYZ cases, `nearby-18` starts in a self-collision/clearance
violation. Numerical multiseed and NN+PRM each solve all **23 eligible starts**.
Existing seed+DLS also solves those 23; the conditional NN alone solves none,
and conditional-rollout refinement without fallback solves 15. All successful
held-out XYZ plans are direct. The separate fixed-endpoint self-collision fixture
requires a detour: its direct connection is rejected, while PRM returns a
three-waypoint validated path of **4.662 rad**. This is evidence of routing under
the URDF convex proxy model, not physical robot validation.

On the same eligible pose5 cases, existing bounded DLS solves **18/23**, while
numerical multiseed IK with direct planning and with PRM both solve **23/23**.
That improvement is due to IK seed/refinement coverage rather than path routing.

Errors average available evaluated endpoint errors; failures without an endpoint
are omitted from that error mean and remain visible in status/success counts.
Times, attempt reasons, eligible-start counts and model/checker metadata are saved
in `artifacts/benchmark_prm_self.json`. The pose5 self-collision comparison is saved
separately in `artifacts/benchmark_prm_self_pose5.json`.
