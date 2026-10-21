# Mouse-aim performance investigation — 2026-10-02

The subsequent [IK recovery upgrade](IK_RECOVERY.md) preserves the normal search
and adds bounded coverage only after IK/routing failure. Its robustness tradeoff
is measured separately: existing successful-query median time stays essentially
unchanged; failed cursor queries spend a median additional 462 ms on recovery.

The baseline's main warm-query bottleneck was numerical 5D IK, especially
repeated FK, Jacobian and directed-axis calculations. NN inference was only
approximately 0.26 ms. Cold first use had a separate 49.35-second roadmap
validation cost. Every warm query also spent approximately 66 ms rebuilding
planning-space metadata, principally by hashing the same mesh files twice.

After measuring these costs, the first optimization pass cached unchanged
geometry fingerprints and reduced repeated small-array FK/Jacobian/axis work.
Nontrivial warm queries are approximately 1.9–2.1 times faster in the selected
cases. Solver selection, collision policy, four-goal budget, checkpoint and
roadmap remain unchanged. No hardware motion was executed.

## Baseline measurements

Machine: Intel Core i9-13900H, 20 logical CPUs, RTX 4070 Laptop GPU with 8 GiB
VRAM. Python 3.12.3, NumPy 2.5.3, SciPy 1.18.1, python-fcl 0.7.0.11,
trimesh 5.1.0. Profiling used `OPENBLAS_NUM_THREADS=1`, matching the UI launcher.
The current Seed5 inference implementation uses NumPy on the CPU.

The four fixed mouse snapshots each had three unprofiled repetitions. The
blocked-route fixture uses known joint endpoints and measures routing plus
trajectory validation separately from IK. Other desktop applications were
running; these diagnostic cases are not a workspace-wide latency benchmark.

| Warm task | Service wall median (ms) | IK (ms) | Routing + trajectory (ms) | Other setup (ms) | HTTP wall median (ms) |
|---|---:|---:|---:|---:|---:|
| Already at target | 78.1 | 3.0 | 5.2 | 69.8 | 100.3 |
| Nearby target, direct route | 398.7 | 313.2 | 15.9 | 69.9 | 423.3 |
| Recovered edge case, direct route | 790.9 | 620.9 | 99.2 | 69.6 | 810.9 |
| Failed 5D search | 2,226.9 | 2,159.2 | 0 | 67.7 | 2,284.2 |

Individual columns are medians and need not sum exactly. NN inference inside
the three nontrivial IK queries was 0.25–0.26 ms. JSON serialization was
0.24–1.02 ms for responses of 8–35 KiB. HTTP timings include server scheduling,
setup, JSON and transport, and were measured after the server was warm.

The first query in a fresh profiling process took **52.56 s**: **49.35 s**
loading/revalidating the roadmap, **2.81 s** constructing the collision space
(including **2.73 s** initializing collision meshes), and approximately
**0.39 s** for IK/planning. Python/module imports were measured separately.

The cached graph contains 400 nodes and 3,003 edges. Its successful cold edge
validation entails 130,378 state-validity calls, plus 400 node checks. The count
is calculated from the existing interpolation formula and saved edge endpoints;
it includes repeated endpoint checks and is not a sampled profiler estimate.

## Code and hotspot evidence

| Finding | Evidence in the current implementation |
|---|---|
| All cached edges are checked on first use | [`Roadmap.load`](prm.py#L261), including the edge loop at line 276; `JointSpace.edge_valid` at line 166 checks endpoints and every interpolated state. |
| A cache hit still constructs the planning space | [`service.plan`](service.py#L57) calls `planning_space` before the roadmap-cache lookup. |
| Two geometry fingerprint requests per planning space | [`checker`](self_collision.py#L163) and [`JointSpace.__init__`](prm.py#L101) both call [`geometry_digest`](prm.py#L82). Before caching, each request read eight meshes totaling 30,237,648 bytes. The new event/stat cache avoids those repeated reads/hashes. |
| Multiple complete numerical solves dominate warm IK | [`generate_goals`](planning.py#L107), seed loop at line 194 and method loop at line 204; [`pose5.solve`](pose5.py#L111). |
| Small NumPy operations were repeated in every iteration | The baseline used six Jacobian cross-product calls plus three direction calls. [`Arm.transform`](kinematics.py#L75) now batches the Jacobian operation and caches rotation coefficients; [`axis_error`](pose5.py#L96) now uses a specialized three-component cross product. |
| Collision checks dominate an obstructed route | [`SelfCollisionChecker._check`](self_collision.py#L124) recomputes link FK, transformed boxes and pair checks; [`trajectory`](planning.py#L225) checks path and interpolated intervals again. |
| Displayed compute time previously omitted setup | [`mouse-aim.js`](../webui/src/mouse-aim.js#L184) now uses `server_compute_ms` from [`service.py`](service.py#L130), which includes geometry setup and roadmap loading. It retains the old-field fallback for older responses. |

The nearby target required six refinements and 416 optimizer iterations; the
edge case required six and 829. The failed search exhausted 26 refinements
(DLS/bounded recovery over 13 seeds) and 2,647 iterations. Refinement alone took
approximately 294, 602 and 2,097 ms respectively in the unprofiled runs. These
iterations are computations, not executed robot motion.

Instrumented profiles attribute roughly 54–58% of refinement time to FK/Jacobian
calls; axis-error calculations and bounded-solver work account for much of the
remainder. The failed-case profile recorded 35,179 `np.cross` calls. General
NumPy axis dispatch/allocation is prominent alongside actual arithmetic.
Inclusive profile times overlap and must not be summed.

The blocked joint route succeeds through PRM and takes a median **797.5 ms**
including trajectory validation. Its profile records 1,927 collision queries;
collision checking occupies approximately 89% of instrumented route/trajectory
wall time. This is a different bottleneck from the direct-route mouse cases.

Warm microbenchmark medians: plain FK **0.126 ms**, FK with spatial Jacobian
**0.302 ms**, state validity **0.334 ms**, complete warm NN proposal call
**0.264 ms**, and planning-space construction **66.18 ms**.

## Optimization options and recommended order

The geometry cache and small-array math changes below have been implemented;
compiled backends, alternate solvers/planners and GPU work remain candidates.

1. **Implemented: cache geometry fingerprints with asset-change
   invalidation.** Compute each content digest once for an unchanged asset set,
   retaining the geometry/environment/clearance key and all path checks. The
   new [`AssetWatch`](asset_watch.py) combines Linux filesystem events with stat
   identity checks; rapid same-size edits, restored mtimes, symlink/replacement
   changes and edits during hashing are covered. Watcher failure falls back to
   fresh content hashing. This reduced warm planning-space construction from
   66.18 to 2.87 ms. Planning metadata is still constructed and checked.
2. **Implemented: optimize FK/Jacobian and axis math; next compare compiled
   CPU kernels.** Constant Rodrigues coefficients are precomputed, six Jacobian
   cross products are batched, unused Jacobian work is skipped for plain FK,
   and direction math uses a validated three-component cross product. FK with
   Jacobian fell from 0.302 to 0.116 ms. Repeated target normalization remains a
   possible next improvement.
   [Numba nopython kernels](https://numba.readthedocs.io/en/stable/user/performance-tips.html)
   can compile numerical loops; [Cython typed memoryviews](https://docs.cython.org/en/latest/src/userguide/memoryviews.html)
   are an alternative. Keep float64 and finite checks; do not enable relaxed
   `fastmath` as part of a semantics-preserving comparison. Numba is not installed;
   its [documented compatibility table](https://numba.readthedocs.io/en/stable/user/installing.html#version-support-information)
   lists current 0.68.0 support for Python 3.12 and NumPy 2.5. Prototype in an
   isolated environment instead of changing the working project's dependencies.
3. **Compare Pinocchio as a native FK/Jacobian backend.** Its
   [`computeJointJacobians`](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/namespacepinocchio.html)
   can compute FK and the model Jacobian together. Reuse our 5D residual and
   acceptance logic. Explicitly verify `g_base`, `joint6_flange`, meter/radian
   units, joint ordering, fixed offsets and a flange-centered, base-aligned
   Jacobian; a spatial WORLD-origin Jacobian has different linear components.
4. **Reduce repeated collision computation.** Try exact-configuration validity
   memoization scoped to an immutable planning profile and query, plus batched
   link FK/AABB calculations. Cache exact values, not rounded configurations.
   Keep the existing clearance, excluded-pair rules, interpolation resolution
   and final trajectory validation. This chiefly helps PRM and longer paths.
5. **Improve numerical iteration efficiency and obsolete-request handling.**
   Benchmark adaptive damping/trust-region steps independently of trajectory
   playback speed. SciPy already provides a bounded
   [`least_squares` trust-region solver with callable Jacobians](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html),
   but it would be an experimental controller requiring equivalent 5D and
   opposite-direction checks. Add cooperative cancellation for superseded
   mouse snapshots: browser abort currently stops delivery, while the server's
   synchronous computation has no cancellation hook. The UI also intentionally
   waits 200 ms before replanning and 750 ms after a busy response; these are
   responsiveness policies, not numerical computation time.
6. **Use the GPU for batched refinements, if compiled CPU work is insufficient.**
   Parallelize seeds/FK/Jacobians and retain logical NN → current → roadmap →
   random admission order. The installed PyTorch stack is a candidate;
   [CUDA graphs](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html)
   reduce launch overhead, which matters for small batches. Alternatively,
   [cuRobo demonstrates GPU multi-seed IK with CUDA graphs](https://curobo.org/get_started/2a_python_examples.html).
   Its standard examples use full position/quaternion goals, so it is not a
   drop-in replacement for our directed local-Z/free-spin task or convex-FCL
   collision assumptions. Preserve the project's final FK/collision/route checks.
7. **Address cold readiness separately.** Background prevalidation plus a visible
   readiness indicator can remove the first-click pause but does not eliminate
   the computation. A [LazyPRM strategy](https://ompl.kavrakilab.org/planners.html)
   checks graph validity on candidate paths and can avoid validating unused
   edges. Adopting it requires an explicit cache-validation policy change;
   metadata, structure and every selected motion still need validation.

Moving only the Seed NN to GPU is low priority: even removing its measured
inference time entirely saves less than 0.1% of these nontrivial query times.
Changing the heap/search algorithm is also unlikely to solve the measured
collision-heavy PRM cost. Preserve the default four-goal budget, all four raw
checks, fallback/seed order, tolerances and collision criteria during the first
optimization comparisons.

## Before/after validation

| Same warm task | Before median (ms) | After median (ms) | Speedup |
|---|---:|---:|---:|
| Already at target | 78.1 | 11.3 | 6.89× |
| Nearby direct target | 398.7 | 189.6 | 2.10× |
| Recovered edge case | 790.9 | 405.3 | 1.95× |
| Failed search | 2,226.9 | 1,165.7 | 1.91× |

The cold query measured 52.56 → 44.95 seconds, and the blocked joint route
797.5 → 678.2 ms. Cold validation remains the largest first-use cost. Cold
measurements were single runs, and the optimized run overlapped the first
simulation/HTTP regression checks; do not interpret the difference as a precise
isolated cold-load speedup.

All four tasks retained exactly the same status, accepted-goal count,
refinement/iteration count, frame count and planning outcome across the three
repetitions. The 75-case sweep retained **56 accepted IK tasks and 56 validated
paths**. Comparing each row with the pre-optimization sweep found no changes
in status, accepted-goal count or planning outcome. All successful measurements
retain actual cursor acceptance and self-collision validation.

The seven new regression checks cover FK/Jacobian/link-frame equivalence across
103 poses, direction equivalence including antipodal and extreme-scale vectors,
unchanged-file cache hits, rapid same-size changes with restored timestamps,
asset-list/symlink changes, deletion, edits while reading, and watcher failure.
The existing shortcut check also verifies that the new server compute timer
includes IK and planning.

Results: [before/after comparison](../output/performance/mouse_aim_optimized/comparison.json),
[optimized profiles](../output/performance/mouse_aim_optimized/profile.json),
[optimized 75-case sweep](../output/performance/mouse_aim_optimized/sweep.json).

The restarted production server's warm HTTP medians are **17.7, 191.9, 419.2
and 1,190.6 ms** for the same four cases, after an unmeasured warmup query.
Baseline HTTP medians were 100.3, 423.3, 810.9 and 2,284.2 ms. See
[optimized HTTP results](../output/performance/mouse_aim_optimized/http.json).
The production build passed, and the running server remains hardware-disabled,
disconnected and idle. SHA-256 checks confirmed both the selected 5D
checkpoint/configuration and all five original XYZ assets are unchanged.

Changed production files: `kinematics.py`, `pose5.py`, `prm.py`, new
`asset_watch.py`, `service.py`, and `webui/src/mouse-aim.js`. The diagnostic tool,
new equivalence/invalidation checks, timer assertion, documentation and a
short-playback browser assertion accompany those changes.

Validation: **164 checks passed** (96 simulation, 45 bridge/HTTP and 23 browser
checks). The first combined browser run passed 22 checks and missed the brief
Pause label on a 0.65-second path played at 4× speed. That existing assertion
now captures the real click-handler state before the next animation frame;
all five affected PRM browser checks passed on rerun. Playback code is unchanged.

## Reproduce and inspect

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.profile_mouse_aim
# With the simulation-only server running on port 8000:
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.profile_mouse_aim --http-only
# Keep a separate comparison directory:
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.profile_mouse_aim --output output/performance/new_run
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.benchmark_mouse_aim --output output/performance/new_run/sweep.json
```

The first command runs in a fresh process and includes full cold roadmap
validation. `--repeats` defaults to three. Successful query measurements assert
the self-collision validation tag and actual cursor gap/direction acceptance.
The HTTP command checks that the server is idle, disconnected and hardware
disabled, and runs one unmeasured warmup query before collecting timed samples.

Artifacts: [wall times and stage profiles](../output/performance/mouse_aim/profile.json),
[HTTP timings](../output/performance/mouse_aim/http.json), plus per-case `.pstats`
and readable `.txt` files under `output/performance/mouse_aim/`.
The reusable diagnostic tool is [`profile_mouse_aim.py`](profile_mouse_aim.py).
Unprofiled wall measurements establish latency; instrumented results establish
hotspots. [Python's profiler documentation](https://docs.python.org/3.12/library/profile.html)
explains why profiler times should not be used as benchmark speedups.

For further optimizations, compare the existing 75-case fixture and
held-out FK tasks for accepted goals, actual cursor errors, route outcomes and
warm latency. Retain XYZ/checkpoint compatibility and the existing simulation,
HTTP and browser regressions. Record both server wall time and browser round-trip
time so setup costs cannot disappear from the displayed timing.
