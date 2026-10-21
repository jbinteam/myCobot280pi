# Reachable disk with a real `ik_failed` result

`ik_failed_reachable.json` is a self-contained simulation fixture. The normal
`classical_tracking.plan(request)` returns `ik_failed` with seed 42 and unchanged
production budgets. There are no mocks, forced failures, or reduced IK budgets.

The object is at `[-255.469080, -61.792557, 262.984924]` mm in `g_base`.
The frozen cone section is at an axial distance of 330 mm, with a 30° half-angle
and a 190.526 mm disk radius. Use the full-precision request in the JSON.
The start is about 128.905 mm from the object, so this fixture exercises an
outward reposition to the disk rather than a normal 20 mm inward approach step.

The planner considers the first eight candidate points. Four fail the physical
reach bound; the other four exhaust both IK passes, giving eight failed solver
jobs and zero route attempts. Candidate 12 (zero-based index 11) is outside that
set. Real IK finds a visible, collision-free pose for it. The saved witness is
within 0.766 mm of the plane and points at the object within 0.074°.

The fixture also includes a complete RRT* geometric route from the exact failed
request's start to that witness. It was generated separately with seed 42 and
800 iterations available; the first solution appeared at iteration 464.
Replay checks joint limits, self/table collision and point-object visibility at
0.005 rad spacing. This proves both endpoint feasibility and a sampled valid
route under this model. The witness is never supplied to the failing planner.
This does not establish that each sampled point is physically unreachable.

To inspect this case in the WebUI, start `./start_webui.sh --simulation-only`
and open `http://127.0.0.1:8000/?case=ik-failed-reachable`. The case loads the
exact saved start, orange object, green disk, and blue saved witness route.
Use **Show start**, **Show reachable pose**, or the route-position slider.
**Run normal planner** sends the original request through the simulation API
and holds the preview at the start; it never supplies the witness to the planner.
The slider previews the geometric route without assigning execution timing.
**Close case** returns to the normal tracking controls.

Run the focused tests from the project root:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest simulation.diffusion.tests.test_ik_failed_reachable -v
```

To record two fresh failures, recover the additional endpoint through real IK,
replay the stored route, and regenerate an RRT* route:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/reproduce_ik_failed_reachable.py \
  --replan-witness --output output/ik_failed_reachable_replay.json
```

The output path must be new. The script uses local simulation; no server or
hardware connection is needed. Camera configuration, model checkpoint and scene
metadata are checked against the fixture before accepting the evidence.

The failure assertion characterizes the current candidate-selection limitation.
When improving the planner, retain this fixture and witness, and change that
expectation to a checked successful route. A future solver improvement should
not be reverted merely to preserve `ik_failed`.

Reachability is relative to the modeled geometry and assumed camera FOV. The
route is geometric, not a full timed execution or hardware qualification. The
330 mm distance is axial; an off-center witness can be farther from the object
in Euclidean distance (about 378.197 mm here).

The fixture pins its original 80° × 80° camera. Reproduction explicitly uses
that recorded model, independently of the current Orbbec FOV configuration.
The WebUI shows both camera models; its fresh planner button uses the current
model. The saved route proof does not automatically transfer to a changed FOV.
