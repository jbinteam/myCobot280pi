# Learned numerical IK starting guesses

`ik_seed.py` provides eight **virtual joint guesses** for numerical IK from the
current six joints, target XYZ and directed flange local Z in the base frame.
The network is 12 → 256 → 256 → 48, with tanh hidden activations and absolute
outputs `joint_center + joint_radius * tanh(raw)`. Every proposal spans the full
operating joint box. Existing `simulation/seed5.py`, its delta-output model and
checkpoints remain separate.

The network never moves the arm and never supplies an accepted raw endpoint.
The existing current-pose bounded DLS attempt runs first. If it fails, enabled
learned guesses replace up to eight random restarts within the same configured
legacy budget. They are ranked by the existing directed FK residual plus a tiny
normalized travel term, since best-of-eight training gives no head-order meaning.
All guesses go through bounded DLS; the robust least-squares fallback also includes
them while retaining its 32-start/160-function-evaluation limits. Every endpoint
must still satisfy finite values, operating bounds, position ≤1 mm, directed
tool-Z error ≤1°, and `ArmEnv.valid` with the live scene. Opposite tool axes fail.
The selected endpoint then enters the existing v2 trajectory planner and all its
path checks. A useful endpoint guess does not establish a collision-free path.

The raw solver API defaults off. The v2 WebUI launchers enable this measured
checkpoint by default with `MYCOBOT_IK_SEED=${MYCOBOT_IK_SEED:-1}`; set
`MYCOBOT_IK_SEED=0` before launch to disable it. For standalone use,
enable with `MYCOBOT_IK_SEED=1` or pass `learned=True` to
`ArmEnv.inverse_kinematics`. Disable explicitly with `learned=False`.
`MYCOBOT_IK_SEED_MODEL` overrides the default
`simulation/artifacts/ik_seed_v1/models.json`. Missing, malformed, nonfinite,
untrained, or incompatible checkpoints supply no guesses and retain the numerical
starts. The checkpoint binds the feature schema, normalization, FK URDF hash,
frames, limits, architecture and completed training. Serving uses NumPy only;
PyTorch/CUDA are training dependencies. Diagnostics identify model status,
proposal/refinement counts, inference latency, and success attributable to a
learned start. Successful current-pose attempts skip inference.

## Actual dataset and training

`simulation/artifacts/ik_seed_v1/dataset.npz` contains 16,000 train target groups,
512 validation groups, and 256 held-out test groups. Each target has two independent
current-pose pairings: 32,000 / 1,024 / 512 rows. Collision-valid full-range joint
witnesses generate target XYZ and directed tool-Z via the shared FK. Witnesses and
current-pose pools are checked with runtime conservative self collision and the
table at 1 mm clearance, with no extra box obstacle. Current pools are independent
between splits. Targets are split **before** pairing; neither target ID nor exact
XYZ/tool-Z label crosses splits. Loading verifies the content hash, FK labels,
joint limits and split separation; optional/full-training scene auditing checks
every distinct stored current and witness again with `ArmEnv.valid`.

FK witness labels are known-reachable configurations and therefore perfect useful
initializations for their targets. They are not claims that random starts solve
those targets. The separate `start_labels.npz` audit uses 64 training target groups
and nine initializations each: witness, paired current and seven random joint
guesses. Exact bounded least-squares refinement and runtime gates produced 249
successful initial-start/final-endpoint labels and 327 rejected starts. This audit
is stored as extra evidence and was **not** used to train/select this checkpoint.

The actual CUDA training used seed 202610072, batch 512 and 8,000 optimizer steps
in 44.45 seconds. It minimizes best-of-eight directed FK task cost, separate
best-of-eight witness supervision (weight 0.05, q1–q5 only), and a tiny normalized
travel term. Random witness q6 spin cannot be inferred from XYZ/tool-Z and is
excluded from the witness penalty. Only validation loss selects weights; test
targets are never scored by the trainer. The selected checkpoint is step 8,000.
Validation pose-best raw guesses average 20.8 mm and 5.89° error; zero raw guesses
meet both strict tolerances. These are approximate numerical initializations,
and improvement must be assessed **after refinement**, against equal budgets.

`dataset.json`, `training.json`, the NumPy JSON checkpoint and a PyTorch checkpoint
record generation seeds, units, robot/geometry/scene metadata, completed steps,
objective and validation history. The held-out benchmark separately records model
hashes and per-case strict endpoint results. Timing from a single corpus pass is
descriptive, not a controlled production speedup claim.

The frozen-model held-out endpoint comparison used 128 shuffled test pairings and
the same 32 bounded-DLS plus 32 robust fallback budget. Both baseline and learned
search passed all 128 strict endpoints; 73 learned runs solved from a learned
initial guess. Observed median latency was 98.1 → 51.5 ms, p95 was 614.8 → 239.0 ms,
and total latency was 20.43 → 10.04 s. Median paired ratio was 1.096 because many
current-first cases need no learned inference. These measurements come from one
alternating paired pass and are descriptive. The 18-case historical regression
corpus remained 17/18 for both methods; the original failed screenshot target
remained a bounded-search miss. Evidence is stored in
`output/learned_ik_seed_20261007/{evaluation,regression}.json`.
Four full v2 trajectory pairs across three separate validation target groups
(indices 0, 1, 6 and 8) passed all endpoint, diffusion path and smoothing gates
for both baseline and learned starts. All four learned runs actually refined a
learned initial guess. This small compatibility check supports launcher activation;
it does not establish a general trajectory-success guarantee or a measured
end-to-end planner speedup. Raw solver enablement remains explicit so dataset
generation and unrelated callers retain their configured behavior.

## Reproduce offline

Run from the project root with the existing training/collision environment:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.diffusion.ik_seed_data
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m simulation.diffusion.train_ik_seed --device cuda --steps 8000
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest simulation.diffusion.tests.test_ik_seed simulation.diffusion.tests.test_ik_seed_data
```

Dataset generation and training perform no robot or network IO. Existing pinned
v2 diffusion weights, robot geometry, operating bounds and acceptance/path
tolerances are retained. Training captures no external obstacles; runtime scene
checks remain authoritative for any configured box.
