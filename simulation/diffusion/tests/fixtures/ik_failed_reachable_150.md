# Replacement failure case with the actual 150 mm finish checked

## Current problem and orientation relaxation

This fixture is the target regression: without a saved hint, reach the actual
150 mm final disk with checked motion while retaining joint limits, collision,
configured camera visibility, timing and cancellation gates. Strict pointing
mode keeps the recorded `ik_failed` limitation. The default approach policy is
now `orientation_mode: strict_then_fov`: retain any checked strict motion, and
run the camera-FOV solver only after strict search returns `ik_failed` or
`no_path`. Explicit `camera_fov` solves the same request without expanding its
eight candidate points or supplying the witness. The hybrid adds the failed
strict pass's processing time before that fallback.

The first positional-IK experiment reached the disk center with 17.8° tool
pointing error, horizontal/vertical image angles 11.55°/13.90°, and 20.10°
minimum view margin. It had a direct 5.18 s checked route. Normal route selection
then found a shorter direct route of about 3.42 s, ending off-center with 32.83°
pointing error while remaining within the rectangular 82° × 68° image bounds
(minimum view margin 10.25°). Those errors exceed the old pointing tolerances,
but all positional disk and path checks still pass.

The live WebUI approach controller completed `[170, 150]` and held at the final
stage with fresh FK and a rectangular camera-arrival check. Evidence, including
the actual no-hint requests, is in `output/orientation_relaxation_20261015/`.
In the case viewer, **Allow camera-FOV fallback if strict planning fails** is
enabled by default; click **Run approach on this case**. Normal demos expose the
same option. It is captured for the run; changing it stops the current approach.
Uncheck it to compare strict-only behavior. **Run Learned IK + RRT*** in
the saved-case inspector still reproduces the strict baseline.

The FOV solver uses bounded XYZ search followed by hard endpoint visibility
and collision acceptance. Learned virtual starts remain available after the
current-pose attempt. The same direct/RRT*, timed-curve and whole-route visibility
validation gates protect motion. This evidence is for the modeled point object
and nominal/assumed camera setup, not calibrated hardware or whole-object visibility.

This is a complete **170 → 150 mm** approach task with a fixed object and the
configured nominal Orbbec Gemini 305 color FOV **82° × 68°**. It keeps the difficult
starting joints from the older 270 → 250 mm diagnostic, but moves the object
100 mm toward the EEF along its initial gaze. The previous object could not
support a 150 mm finish; this one has an independently checked final endpoint,
a complete collision/visibility-valid route, and checked timing for the full route.

The exact updated WebUI controller, replayed against the live simulation planner,
first requests and reaches 170 mm. Its next request is the final 150 mm disk.
The normal planner returns `ik_failed`: eight points pass the physical reach
filter, sixteen IK jobs exhaust, and no endpoint or route attempt is produced.
No failures were injected and production planner budgets were not reduced.

Candidate 10 in the exact recorded controller request, outside the first eight,
provides a visible collision-free endpoint. Its final axial error is 0.639 mm,
radial distance 85.451 mm within the 86.603 mm disk radius, and pointing error
0.136°. **150 mm is the disk's axial distance**; the off-center witness is
173.187 mm from the object in a straight line. It passes the actual frontend
final-stage arrival gate and completes the two-stage schedule.

The separately generated geometric route has nine waypoints. Dense replay at
0.005 rad checks 1,246 samples with no collisions, occlusions or out-of-frame
samples; the minimum view margin is 4.012°. A fresh RRT* search also regenerated
a valid route to that same witness.

The complete saved route was additionally divided into bounded at-rest
references on the same geometric edges and checked using the production shared
velocity profile and independent reference validator. All **28 references /
6,999 frames**, including the final endpoint, pass joint limits, motion bounds,
collision and point-visibility replay. The full reference duration is about
139.96 s; this proves feasibility rather than an efficient route. None of this
proof supplies a saved goal or route to the failing planner.

This replaces the earlier diagnostic for work on a completable 150 mm approach.
The strict-only comparison still fails the recorded attempt. The default hybrid
approach reaches the finish by invoking camera-FOV fallback after that failure;
it never receives the independently saved witness.
Visibility uses a point object, nominal FOV and an assumed identity mount.
Live calibration, rounded encoder checks, dynamics and physical execution
remain unqualified. No hardware IO was performed.

Open `http://127.0.0.1:8000/?case=ik-failed-reachable-150` in the running WebUI
(or start `./start_webui.sh --simulation-only`). Gray shows the initial 170 mm
disk, green the final 150 mm disk, and blue the independent saved witness route.
**Run approach on this case** uses only the saved initial setup and the normal
planner; it holds/retries on failure. **Show reachable pose** and the route slider
inspect the checked proof. The old diagnostic remains separately labeled in
the case selector.

Reproduce and independently verify the fixture (choose a new output path):

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/reproduce_ik_failed_reachable.py \
  --fixture simulation/diffusion/tests/fixtures/ik_failed_reachable_150.json \
  --replan-witness --output output/reachable_150_replay.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest \
  simulation.diffusion.tests.test_ik_failed_reachable.Final150ReachableIKFailureTest -v
```

Evidence is in `output/approach_150_reachable_failure_20261015/`: controller
requests/results in `approach_replay.json`, geometric proof in `validation.json`,
and complete timed proof in `timed_route.json`. The fixture embeds all timed
references for independent reconstruction. `final_validation.json` repeats
the failures and verifies the exact final fixture, including every timed reference.
When the planner improves, preserve
the proof and update the expected failure to checked successful motion.
