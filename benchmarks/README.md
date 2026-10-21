# Offline planner benchmark suite

Run repeatable correctness and timing comparisons of the current pinned v2 planner with eager neural inference and CUDA graph inference. The runner uses isolated offline workers and never sends robot commands. It does not start the WebUI, reconnect the Pi, run training, or qualify physical execution.

The October 6 snapshot referenced by [`current.json`](current.json), its `fixed-v2-velocity-profile-20261006-190739` directory, and the backup archive are historical immutable artifacts. This suite does not replace or promote them. Both comparison backends use the current source and collision implementation; the comparison isolates neural replay, not the combined improvement over the archived implementation.

## Commands

From the project root, use the project's existing virtual environment and CUDA installation:

```bash
# Inspect the CLI before running a benchmark.
.venv/bin/python -m benchmarks.suite --help
.venv/bin/python -m benchmarks.suite --mode full --list-cases

# Quick: nine cases, seed 42, one ABBA block per case.
# Output must be a new path; choose a new name for each run.
.venv/bin/python -m benchmarks.suite --mode quick --repeats 1 --seeds 42 --output output/benchmark_quick_001

# Full: 26 cases, two seeds, two ABBA blocks per case.
.venv/bin/python -m benchmarks.suite --mode full --repeats 2 --seeds 42 43 --allow-long-run --output output/benchmark_full_001

# Reliable timing: three representative successes, seed 42, ten ABBA blocks.
# 20 measured trials per backend/case; requires several minutes of GPU time.
.venv/bin/python -m benchmarks.suite --mode timing --allow-long-run --output output/benchmark_timing_001

# Narrow a timing investigation; fewer than ten blocks stays inconclusive.
.venv/bin/python -m benchmarks.suite --mode timing --cases recent_live_start --repeats 2 --allow-long-run --output output/benchmark_timing_diagnostic_001

# CPU-only schema/FK/collision fixture validation; no neural planning.
.venv/bin/python -m unittest benchmarks.tests.test_scenarios -v
```

`--timeout 180` bounds worker waiting. An interrupted, timed-out, or incomplete run is not a passing comparison. Full and timing evaluation require the explicit `--allow-long-run` flag because repeated long-motion cases can consume substantial GPU time. The runner records model/environment/source identity and outcomes with the report. Retain the complete output directory, not just the HTML view.

The output includes `report.json` for machine-readable comparison and `report.html` for review, with raw worker records retained alongside them. Warmups and cold setup are separate from warm measurements. Each ABBA block schedules eager → CUDA graph → CUDA graph → eager, giving two measured trials per backend. Timing is instrumented offline planning latency, not end-to-end UI or network latency. GPU load and temperature help explain environmental interference; they do not correct for it.

## Reliable timing

Timing mode defaults to `recent_live_start`, `heldout_000062_block`, and `heldout_000590_block`, all at historically qualified seed 42. These cover the offline encoder-preparation route plus shorter and longer blocked planning. Each of ten ABBA blocks uses four fresh sequential processes, one complete warmup followed by one measured plan per case. That yields **120 measured plans plus 120 warmups**, with 20 measurements per backend/case. `--cases` selects named cases from the mode's catalog; timing uses the full catalog. This subset is not a new correctness qualification of all 26 cases.

Child processes pin `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` to 1 before imports. Actual PyTorch thread counts are recorded separately; the production policy's Torch settings remain intact. CUDA synchronizes before and after the measured plan. Setup, graph capture/warmup and encoder preparation remain separate. Both backends still include stage probes and prediction hashing in the generation measurement, so these are **instrumented offline timings**, not UI latency or uninstrumented planner latency.

Each complete block contributes one paired ratio: mean of its two eager times divided by mean of its two graph times. The report uses the median block ratio and a deterministic 10,000-resample percentile bootstrap interval over **whole blocks**. It never treats the two measurements in one backend/block as independent resampling units. The approximate per-case 95% interval appears only with at least ten blocks. It is exploratory, is not adjusted for multiple cases, and can understate uncertainty when neighboring blocks remain correlated.

Correctness and timing support are separate. A timing-mode speedup is supported only when all requested trials pass the existing exact-output gates and converge, at least ten complete blocks exist, the interval's lower bound exceeds 1×, GPU telemetry is usable with no competing compute process observed, and neither variation rule triggers:

- Backend IQR divided by median latency exceeds 10%.
- First-half versus second-half median block ratio changes by more than 10%.

Those thresholds are declared operational checks, not statistical proofs of stability. Otherwise timing is **inconclusive**, with reasons and all samples retained. Expected rejections have no speedup claim. No outliers or contaminated blocks are silently discarded. Quick/full modes remain descriptive smoke/qualification measurements, even if correctness passes.

Parent-side GPU polling records UUID, clocks, temperature, utilization, power and compute PIDs around worker activity. Per-case summaries cover warmup plus measurement. Missing telemetry or observed foreign compute prevents a supported claim; monitoring never stops another process. Approximately one-second sampling can miss shorter competing work, and monitoring itself adds system overhead. Clock/temperature ranges help interpretation; they do not normalize the results or prove absence of throttling. Keep the raw worker and telemetry files alongside the report.

The warmup, synchronization and thread controls follow [PyTorch's benchmark guidance](https://docs.pytorch.org/docs/stable/benchmark_utils). The interval uses the [percentile bootstrap procedure](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html), implemented on block ratios without adding SciPy to the report-only path. Results apply to the selected scenarios, seeds, instrumented workload and recorded machine configuration.

## Scenarios and conventions

[`suite/cases.json`](suite/cases.json) freezes every input, including source provenance and the archived encoder mapping used for one offline profile case. Loading the manifest does not import Torch, create a simulation scene, or access the robot.

| Group | Count | Qualification |
|---|---:|---|
| Three existing held-out examples, each with and without its block | 6 | Saved successful seed-42 planning runs |
| Archived recent live-start request | 1 | Saved successful seed-42 planning run; offline encoder preparation |
| Small moves | 3 | Exact FK goals and valid start/end geometry; convergence exploratory |
| Flange direction changes | 3 | Exact FK goals and valid start/end geometry; convergence exploratory |
| Reversed held-out endpoints | 3 | Longer moves; valid start/end geometry; convergence exploratory |
| Near operating limits | 5 | Goal one degree inside a selected bound; convergence exploratory |
| Self/table/block start rejection, blocked goal, unreachable target | 5 | Explicit expected rejection status |

Quick mode selects the seven saved-success cases plus block-start rejection and an unreachable target. Full mode selects all 26. Seeds beyond 42 do not inherit historical success qualification: the seven saved-success cases become explicitly exploratory for those seeds. The rejected starts and blocked goal are validated with the actual CPU collision scene; the unreachable target exceeds a conservative URDF-chain reach bound while remaining inside the API's coordinate input bounds.

All 14 new exploratory cases supply an exact `goal_deg`, so they bypass endpoint IK and test rollout toward a known joint goal. The seven historical cases and unreachable-target case exercise Cartesian-to-IK selection. Each case records `endpoint_selection` to keep these workloads distinct. Default quick mode at seed 42 contains no exploratory cases; using another seed, such as `--seeds 43`, makes its seven historical cases exploratory.

Every request explicitly records:

- Joint angles in degrees; target XYZ in millimeters in `g_base`.
- Directed `joint6_flange` local Z axis, with local Z spin free and no added tool transform.
- An absent block (`null`) or block center XYZ and half-extents XYZ in millimeters. Blocks rest on the −35 mm table.
- Seed 42 in the frozen request, four samples, 32 denoising steps, and 200 maximum decisions. The runner records any selected evaluation seed separately.
- Downstream speed 70% and `smooth_trapezoid` profile with 5% rise / 10% fall. These shared settings are applied only to the representative `recent_live_start` offline retiming/encoder preparation case. They do not affect neural trajectory generation; other cases are not retimed for hardware playback.

The historical October 6 benchmark used **5%** speed and original quintic timing. Do not directly compare its retiming/encoder preparation totals with the current **70%, 5/10** setup. The saved October 7 optimization results tested both original quintic and adjustable ramps; this suite's default profile is adjustable ramps only.

## Reading results

Correctness gates require matching outputs between the two implementations, including generated sample fingerprints, returned trajectories and relevant planner results. The representative no-block profile also checks offline encoder preparation. Matching failure statuses alone do not establish a successful plan.

The recorded planner `duration_s` describes the simulation reference trajectory, including in the representative profile case. Its separate encoder-preparation timing measures offline processing cost, not physical execution duration.

`expected.kind = success` requires successful planning. `reject` requires the named rejection status. `allowed` marks unqualified exploratory convergence and permits only `success`, `decision_limit`, or `no_safe_chunk`, with matching baseline/candidate outcomes. Timeouts, exceptions and validation failures are never accepted exploratory outcomes. Successful planning still requires the planner's collision and endpoint checks; merely constructing a reachable FK goal does not prove the planner can find a route.

Timing summaries use successful valid trials. Rejections and nonconverged cases remain visible in correctness results but do not inflate speedup claims. Reports show sample counts and spread; p95 is withheld below 20 samples. Two warm samples per backend are a smoke comparison, not strong evidence of a small performance change. Use more repeats when evaluating small differences and keep other GPU work idle.

Kinematic endpoint errors are derived from the URDF model and joint references. The suite does not measure external Cartesian accuracy, real joint tracking, settling time, command jitter, motor dynamics, payload behavior or real minimum clearance. Historical physical endpoint convergence remains a separate unresolved qualification issue.

## Adding or qualifying cases

1. Add a unique stable lowercase `id`, category, description and complete request to `suite/cases.json`. Keep source evidence or the deterministic construction method. Never copy a saved success label to a changed request.
2. For a new reachable case, use `ARM.transform` to derive both XYZ and the directed flange Z axis from an explicit `goal_deg`. Validate both start and goal in the exact scene. This removes reachability ambiguity but does not qualify route convergence.
3. Initially mark new cases `qualified: false`, `expected.kind: allowed`, with the three bounded statuses above. Add a strict rejection only when its specific cause is checked. The CPU tests verify actual scene geometry and exact FK targets.
4. Run schema and geometry tests, then evaluate chosen seeds. Promote convergence expectations only with retained successful evidence for those exact inputs, model and seeds. Update `scenarios.py` qualification validation and tests deliberately when expanding the qualified seed set.
5. Keep quick mode small and representative. Store outputs in a new directory and retain earlier reports; update neither the frozen October 6 archive nor `current.json` as part of routine benchmarking.

The runner is for implementation regression comparisons. Comparing different planning algorithms requires a different quality contract and cannot rely on exact-output identity alone.

## Collision and remaining CPU profiling

For diagnostic function-level attribution on the three timing cases:

```bash
.venv/bin/python -m benchmarks.suite.profile_cpu --output output/benchmark_cpu_profile_001
```

The default uses CUDA graph inference, seed 42, and two repetitions per case. Use `--cases` to select cases, `--backend eager` to inspect eager inference, and `--repeats` for additional diagnostic repetitions. This is a separate profiling run, not another speedup qualification. Keep its output separate from reliable-timing results.

Each case has a warmup, an unprofiled control, and a profiled execution. The profiler observes only `service.plan`; offline encoder preparation remains outside that profile. Exact prediction, reference, RNG and encoder signatures must match the control, warmup, and selected historical correctness reference. Failed or incomplete diagnostics cannot establish a valid function breakdown. Production collision predicates and method identities remain intact, including the batched-validity dispatch.

The report retains raw `.pstats` files, function call counts, exclusive time, cumulative time and caller relationships. Exclusive function times can be added; cumulative times include descendants and **must not be summed**. Disjoint exclusive categories attribute a function to its defining module, not to its entire call stack. Shared NumPy and native work may therefore remain ambiguous. FCL/Cython internals that the profiler cannot observe may be charged to the enclosing Python function. Profiler timings include observer overhead and CPU waits; they are neither pure CPU execution time nor CUDA kernel duration. The profiled/control ratio mixes profiling overhead with host variation; it does not isolate either effect or establish an implementation speedup. See the [Python profiler documentation](https://docs.python.org/3/library/profile.html).

Parent-side Linux CPU observations record process CPU accounting, per-core host busy counters, main-thread context switches, allowed CPU affinity, sampled frequencies and governors. Scheduler runnable-wait time is reported only when kernel scheduler statistics are already enabled; the benchmark does not enable them or change power settings. CPU rates use the first and last interior observations, so short windows may have insufficient data. Host busy time includes the benchmark itself, process CPU percentages include all its threads, and frequency snapshots cannot prove throttling or reconstruct thread migration. The raw observations and missing-data fields are retained. See the [Linux proc documentation](https://www.kernel.org/doc/html/latest/filesystems/proc.html) and [scheduler statistics](https://www.kernel.org/doc/html/latest/scheduler/sched-stats.html).

## Comparing the scheduler timestep cache

Use the dedicated comparison to isolate the DDPM scheduler's timestep lookup cache. Both baseline (`cache_off`) and candidate (`cache_on`) use CUDA graph neural inference, the same current production source, and unchanged collision checks. Fresh child processes differ only in `MYCOBOT_SCHEDULER_CACHE=0` versus `1`; the report verifies the actual policy flag and cache-hit counters. An enabled cache with no observed hits in a sampled trial fails the comparison.

```bash
# Short correctness comparison; fewer than ten blocks cannot support a speed claim.
.venv/bin/python -m benchmarks.suite.compare_scheduler \
  --cases recent_live_start heldout_000062_block --seeds 42 --repeats 1 \
  --reference output/benchmark-suite/20261007T070319.954279Z/report.json \
  --output output/scheduler_cache_check_001

# Repeated representative timing. CPU 4 must be allowed on this machine.
# Affinity applies only to child workers; parent and OS settings are unchanged.
.venv/bin/python -m benchmarks.suite.compare_scheduler \
  --cases recent_live_start --seeds 42 --repeats 10 --allow-long-run \
  --cpu-affinity 4 \
  --reference output/benchmark-suite/20261007T070319.954279Z/report.json \
  --output output/scheduler_cache_timing_001
```

The default is `recent_live_start`, seed 42 and ten ABBA blocks, requiring `--allow-long-run`. `--cases` accepts any full-catalog case IDs. `--reference` is optional; when supplied, it checks exact requests, model/URDF identity, prior signatures and all production source except the explicitly allowed scheduler-cache patch files (`policy.py`, `service.py`, `timestep_cache.py`, and their new `test_timestep_cache.py` regression test). Historical absolute timings are never mixed into the new comparison.

The output includes raw worker events, GPU telemetry, cache activity, actual child CPU affinity, source/frozen-artifact hashes and an HTML/JSON comparison. Timing claims reuse whole-block confidence intervals and all-success/telemetry/stability gates. A passing correctness result with `inconclusive` timing establishes output parity only. Fixed affinity does not fix CPU clocks or eliminate host interference. No physical motion or hardware tracking is qualified.

## Comparing collision joint resets

The collision comparison holds CUDA graph inference and the scheduler cache
on in both modes. `scalar_reset` uses six Bullet reset calls per pose;
`batch_reset` submits the same six positions and zero velocities together.
Both retain every pose, FCL self check and Bullet table/block clearance query.
The worker verifies each scene's actual reset mode and records reset call counts.

```bash
.venv/bin/python -m benchmarks.suite.compare_collision \
  --cases recent_live_start heldout_000590_block --seeds 42 \
  --repeats 10 --allow-long-run --cpu-affinity 4 \
  --output output/collision_reset_timing_001
```

Use `--repeats 1` without `--allow-long-run` for a short correctness comparison.
The default correctness reference is
`output/scheduler_lookup_optimization_20261007/quick/report.json`; only its
`cache_on` rows are accepted. Production changes versus that reference are
restricted to `env.py` and the new `test_bullet_reset.py` regression tests.
Historical times are excluded. The same ABBA confidence, output identity,
telemetry, source integrity and stability gates apply. Workers perform no robot IO.

## Comparing trajectory construction

The trajectory comparison isolates the combined derivative-hull subdivision and
final-output frame FK optimization. `recursive_math` uses the original recursive
bounds and scalar frame FK; `batched_math` batches independent calculations with
the same arithmetic. CUDA graph inference, scheduler caching and Bullet joint
batching remain enabled in both settings.

```bash
.venv/bin/python -m benchmarks.suite.compare_trajectory \
  --cases recent_live_start heldout_000590_block --seeds 42 \
  --repeats 10 --allow-long-run --cpu-affinity 4 \
  --output output/trajectory_construction_timing_001
```

Use `--repeats 1` without `--allow-long-run` for a short correctness comparison.
The default historical reference is
`output/collision_validation_optimization_20261007/quick/report.json`, using its
`batch_reset` rows. The comparison verifies actual trajectory mode, batch work,
matching helper calls/frame counts/joint-reset counts and exact planner outputs.
Only the trajectory helper, bound dispatchers, final-frame caller and their new
regression tests may differ from that reference. The existing ABBA, confidence,
telemetry and source-integrity gates apply; historical timings are excluded.
`MYCOBOT_TRAJECTORY_BATCH` is read at module import, so change it before starting
a new process. No robot IO is performed.
