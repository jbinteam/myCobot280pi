# Guided joint-space repair experiment

Guided sampling improved recovery cost in an empty-roadmap stress test. It did
not improve the existing cached planner on this sample: both the production
400-node graph and a 64-node induced subgraph already solved all 24 tasks.
The UI now enables the fallback for PRM previews and mouse aim through the
**Repair blocked routes** checkbox. Python/API callers opt in explicitly.

## UI integration

The checkbox is enabled on first use and remembers the user's choice. Both
**Plan self-checked PRM** and **Mouse aim** send `guided_route_repair` and a
32-node budget. Normal route search and any existing expansion to additional
IK goals finish first. Only a remaining graph-connection failure permits one
temporary repair attempt. Successful routes return without sampling, and no
repair runs when IK has not produced a valid goal.

The PRM and mouse-aim result panels show whether repair was off, unnecessary,
successful, or unsuccessful, plus its sample count and time when attempted.
Changing the setting clears the preview and discards a late PRM response;
active mouse aim stops. Comparing a captured mouse target retains its repair
setting. Final trajectory and cursor acceptance still gate all playback.
[Hardware mouse aim](../webui/HARDWARE_MOUSE_AIM.md) also uses this fallback
through its separate live-start, server-owned execution path.

`service.plan` and `service.mouse_aim_plan` accept `guided_route_repair`
(boolean, default false) and `route_repair_samples` (integer 1–128, default
32). The UI supplies the opt-in flag. `guided_repair.repair_failed_route`
accepts the existing planner result so the integration does not repeat a
baseline route search. The original `guided_repair.plan` wrapper remains
available for standalone use.

The benchmark results below and `comparison.json` describe the experiment
before UI integration. They remain unchanged for comparison with that state.

UI integration validation covers 60 backend tests and 17 browser tests,
including a real self-collision detour on an empty graph, disabling repair,
late-response rejection, mouse-aim settings, desktop/mobile layout, and final
trajectory gating. Live HTTP checks confirm that the rebuilt UI is served,
the API honors the checkbox flag, and successful cached routes stay unchanged.
See [live API checks](../output/guided_repair/ui_live_smoke.json) and
[integration validation](../output/guided_repair/ui_integration_validation.json).

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest \
  simulation.tests.test_guided_repair_integration \
  simulation.tests.test_guided_repair simulation.tests.test_prm \
  simulation.tests.test_ik_recovery simulation.tests.test_mouse_aim -q
cd webui
npx playwright test tests/prm.spec.js tests/mouse-aim.spec.js
```

## Method

[guided_repair.py](guided_repair.py) adapts the guided/uniform mixture from
Wang et al., *MS-Bi-PRM: A dynamic-ready bidirectional probabilistic roadmap
algorithm with multi-strategy sampling for high-efficiency robotic manipulator
path planning*, Expert Systems With Applications 330 (2026), 133041,
DOI [10.1016/j.eswa.2026.133041](https://doi.org/10.1016/j.eswa.2026.133041).
The local source is
[the research PDF](../research_references/1-s2.0-S0957417426019524-main.pdf).
This implements the sampling idea in bounded six-joint space, rather than
reproducing the paper's complete Cartesian planning algorithm.

The existing planner runs first. Only `start_not_connected`,
`goals_not_connected`, or `disconnected_components` permit one repair attempt.
Existing successes and invalid inputs return immediately.

Each proposal has a 75% chance of being a Gaussian perturbation of an
interpolated start–goal joint vector and a 25% chance of being uniform across
the operating limits. The noise standard deviation is 8% of each joint's
operating range. Gaussian proposals are clipped to those limits; all six
joints are bounded, including the wrist, with no angle wrapping. Sampling
stops at 32 accepted unique nodes by default, or after eight times the node
budget in proposals.

The wrapper copies the reusable graph and adds validated temporary nodes and
edges. Each new node considers at most 12 nearest neighbors in the existing
normalized joint metric. The normal planner retries once on that temporary
graph, including its existing shortcuts and final path checks. No persistent
roadmap or model asset is modified.

## Comparison on 2026-10-02

[benchmark_guided_repair.py](benchmark_guided_repair.py) compares the original
planner with uniform and guided sampling at equal budgets of 32 and 64 accepted
nodes. It uses 20 valid endpoint pairs with a self-collision obstruction on
their direct joint interpolation, plus four unobstructed controls. One blocked
case is the existing self-collision fixture; 19 are independently sampled using
seed 20261002. Every blocked case has a validated full-roadmap route witness,
retained for audit and never supplied to the repair planner. No case was
discarded for lacking that witness.

Known joint goals isolate routing from IK. Goals retain the start's joint-six
angle in the randomly generated cases. Query timing includes routing and
final speed-limited trajectory validation, with the same lightweight validity
and edge counters enabled for every method. Cold roadmap validation (about
26 seconds) and task selection are excluded. Method order is shuffled within
each repetition. Sampling seeds are fixed per case, so repetitions measure
timing variation rather than independent random-seed reliability.

The cached 400-node and 64-node profiles each ran three repetitions per task
and method. All five methods passed all 24 tasks; repair never triggered and
paths and collision-check counts matched their respective baseline. Median
time for the 20 obstructed tasks was 397 ms for the original 400-node planner
and 493 ms for the 64-node planner. Small timing differences between wrappers
are not evidence of improvement when they perform the same work.

To exercise actual recovery, a separate stress profile removes every cached
node while retaining the same tasks and validation settings:

| Empty-roadmap method | Validated blocked routes | Median query time | Median collision queries |
|---|---:|---:|---:|
| Original planner | 0/20 | 6 ms (failed queries) | 24 |
| Uniform, 32 nodes | 19/20 | 4.20 s | 20,281 |
| Guided, 32 nodes | 20/20 | 3.04 s | 14,373 |
| Uniform, 64 nodes | 19/20 | 7.03 s | 33,779 |
| Guided, 64 nodes | 20/20 | 5.12 s | 23,199 |

The first two stress cases retain three repetitions from a pilot; the
remaining 22 tasks have one repetition. Each case contributes equally to the
table: first take its median time/count, then take the median across cases.
The timing columns include failed queries. Every method passed all four
direct controls without triggering repair.

Both uniform budgets failed `obstructed-12` with `goals_not_connected`; both
guided budgets recovered it. On the 19 obstructed tasks where both methods
succeeded, the median paired guided/uniform ratios were:

| Budget | Query time | Collision queries | Normalized joint path cost |
|---|---:|---:|---:|
| 32 | 0.687 (31.3% lower) | 0.681 (31.9% lower) | 0.689 (31.1% lower) |
| 64 | 0.706 (29.4% lower) | 0.693 (30.7% lower) | 0.660 (34.0% lower) |

These are medians of per-case ratios, not ratios of the table's medians.
Path cost sums joint displacement normalized by each joint's operating range;
it is not Cartesian path length. Both guided budgets solved this sample, so
32 nodes is the more economical default for this experimental wrapper.

## Validation and limits

Every accepted route retains the existing 1 mm conservative self-collision
clearance and 0.04 rad edge-check resolution. The benchmark then validates
the complete trajectory at 1.2 rad/s with a 0.05 s frame interval and checks
the 1 mm position and 1 degree directed tool-Z tolerances. The start and final
joint vectors must match the supplied endpoints. Intermediate frames may
change tool-Z direction, as in the existing joint-space planner.

Thirty tests passed: six repair tests plus the existing 24 PRM tests. They
cover route recovery, immutable graphs, bounded deterministic sampling,
finite budgets, invalid configuration settings, unchanged direct successes,
and a disconnected wall that must remain unsolved. Both complete benchmark
reports verify graph immutability and unchanged hashes for all seven original
roadmap, dataset, training, and model assets. The simulation UI remains running
with hardware disabled, disconnected, and idle.

This small development set does not establish workspace-wide reliability.
The empty graph deliberately removes coverage; it is not representative of
normal cached routing. The experiment does not address unreachable IK goals,
external obstacles, physical motion, or continuous collision guarantees.
Guided recovery is substantially slower than the successful warm 400-node
planner here, so these results support a bounded fallback, not a replacement.

## Use and reproduce

Callers supply validated joint goals in radians and validate the returned
trajectory through the existing planning layer before playback:

```python
from simulation.guided_repair import RepairConfig, plan

result = plan(roadmap, start, [goal], RepairConfig(samples=32), shortcut=True)
```

From the project root:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest \
  simulation.tests.test_guided_repair simulation.tests.test_prm -q

OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.benchmark_guided_repair \
  --output output/guided_repair/benchmark.json

OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.benchmark_guided_repair \
  --replay-from output/guided_repair/benchmark.json \
  --sparse-nodes 0 --repair-only --repeats 1 \
  --output output/guided_repair/empty_roadmap.json
```

The last command runs one repetition for every task. The saved stress result
also retains the pilot's completed trials using
`--resume-from output/guided_repair/empty_pilot.json`. Repair paths are
deterministic for these fixed seeds; measured times depend on machine load.

Saved results:

- [Compact comparison](../output/guided_repair/comparison.json)
- [400-node and 64-node trials](../output/guided_repair/benchmark.json)
- [Empty-roadmap trials](../output/guided_repair/empty_roadmap.json)
- [Pilot snapshot](../output/guided_repair/empty_pilot.json)
