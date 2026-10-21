# Reachable IK failure after current-distance stage initialization

**Scope correction:** the failed **250 mm stage** is reachable, but the full
approach to **150 mm** is impossible for this fixed object under the URDF model.
Every EEF position is at least 187.421 mm from the object by the conservative
outer reach sphere. Any 150 mm / 30° disk, even with the 10 mm axial arrival
allowance, is at most 181.934 mm from the object. Those sets cannot intersect
for any disk axis. The WebUI approach run is therefore a diagnostic of the
250 mm stage, not a task capable of finishing at 150 mm. The proof and numerical
bound are in `output/inward_ik_failure_search_20261015/full_approach_reach_check.json`.

`ik_failed_reachable_inward.json` is a second independent simulation case.
It remains applicable after the WebUI approach fix: fresh FK gives an initial
object distance of 270 mm, the 270 mm disk is reached with the current pose,
and the next 250 mm stage returns `ik_failed` with seed 42 and production budgets.
The object remains fixed and visible. No solver failures or reduced budgets are injected.

The first eight candidate points all pass the physical reach bound. All sixteen
IK jobs (two passes over those points) exhaust, leaving zero endpoints and zero
route attempts. Candidate 16 (zero-based index 15), outside that set, yields
a visible collision-free pose. A separate RRT* search finds a complete route
from the exact failed starting joints; the saved route has eleven waypoints.
The geometric witness was candidate 13 during exploration. Replaying the exact
WebUI request changes floating-point tie ordering among equidistant samples;
the saved controller request identifies that same point as candidate 16.
Independent replay at 0.005 rad checks joint limits, self/table collision and
point visibility across 1,132 samples, with minimum view margin 1.286°.

The witness has axial error 0.611 mm, radial distance 142.161 mm within the
144.338 mm disk radius, and pointing error 0.079°. Its Euclidean object distance
is 288.124 mm: the 270 → 250 mm step moves the **disk plane** inward, and the
off-center endpoint and route need not monotonically decrease Euclidean distance.
The object point is not asserted to be inside the arm workspace; the standoff
disk is reachable. This is a geometric route proof under the assumed camera
model, not a full timed execution or hardware qualification.

The actual updated `ApproachSimulation` was replayed against the live simulation
planner. Its first request returned `target_reached` at 270 mm, passed the frontend
path/visibility checks and analytic sampler, and its next request returned
`ik_failed` at 250 mm. The harness stopped after recording that failure. Normal
approach operation would hold and retry with seed 43; this fixture does not
establish permanent failure across retries.

View the case in the running WebUI at
`http://127.0.0.1:8000/?case=ik-failed-reachable-inward` (start with
`./start_webui.sh --simulation-only` if needed). Gray shows the completed 270 mm
disk; green shows the failed 250 mm disk. **Show reachable pose** and the route
slider inspect the saved geometric proof. **Run Learned IK + RRT*** repeats the
failed 250 mm request and holds the preview at the start. The case selector
switches between this scenario and the older outward repositioning case.
**Run approach on this case** starts the real simulation controller from the
saved initial pose with this fixed object and the current configured camera.
It anchors the initial disk at 270 mm, then requests 250, 230, …, 150 mm only
after verified arrival. Failed searches hold and retry. Use the normal
**Pause**, **Resume**, and **Stop approach** controls above the scene. Random
repeats are turned off for this single case run; the saved witness is not supplied
to the controller.

Reproduce two baseline failures, recover the additional candidate through real
IK, densely replay the saved route, and regenerate a route:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/reproduce_ik_failed_reachable.py \
  --fixture simulation/diffusion/tests/fixtures/ik_failed_reachable_inward.json \
  --replan-witness --output output/inward_ik_failure_replay.json
```

The output path must be new. Run both saved cases and the initial-stage check:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest simulation.diffusion.tests.test_ik_failed_reachable -v
```

Discovery and controller evidence are in `output/inward_ik_failure_search_20261015/`:
`found.json`, `approach_replay.json`, `validation.json`, and the exploratory
`search.py`/`search.log`. The deterministic search screened 114 visible inward
requests before discovering this case. Its screen stops when an endpoint is
found, so that count is not a full planner success-rate benchmark.

Keep this fixture and its independent reachability proof when improving the
planner. If the baseline starts succeeding, update the expected behavior to a
checked successful route rather than reverting the improvement.

The fixture pins its original 80° × 80° camera. Reproduction explicitly uses
that recorded model, independently of the current Orbbec FOV configuration.
The WebUI shows both camera models; its fresh planner button uses the current
model. The saved route proof does not automatically transfer to a changed FOV.
