# 5D upgrade results — 2026-10-02

The checkpoint evaluated on 2026-10-02 was full-trained and separate from XYZ. It uses
12 → 384 → 384 → 24 with tanh hidden layers and four bounded joint proposals.
GPU: NVIDIA GeForce RTX 4070 Laptop; PyTorch 2.14.1+cu130. The winning run used
96,000 valid training pairs, 2,048 independent validation pairs, 32,000 steps,
512 examples per batch and seed 72. Validation loss selected step 30,401.
Weights: position 1.0, direction 0.72, travel 0.00001, auxiliary witness 0.03.
Position loss uses squared error / (0.3 m)²; direction uses 1-dot; travel uses rad².
Four full GPU candidates completed 80,000 optimization steps in total.

The WebUI switched to the collision-aware fine-tune on 2026-10-04. The
measurements below describe the previous checkpoint; see
[SEED5.md](SEED5.md#validation-and-measured-results) for collision-training
results and `artifacts/seed5_v1/activation.json` for the active model and backup.


## Validation-only model selection

512 fixed validation tasks; both tolerances 1 mm / 1°, self-collision validity,
one accepted-goal budget, NN refinement only and conventional fallback disabled.
Task success selects the model; refinement iterations break ties. Held-out tasks
were not used for selection. All four models used the same independent validation split.

| Candidate | Training pairs / steps | Valid refined success | Mean refinement iterations | Raw mean errors |
|---|---:|---:|---:|---|
| w256-s42 | 24,000 / 8,000 | 510/512 | 47.7 | 29.44 mm / 9.15° |
| w384-s52 | 24,000 / 16,000 | 510/512 | 103.9 | 23.03 mm / 7.77° |
| w384-s62-witness | 24,000 / 24,000 | 511/512 | 121.7 | 20.66 mm / 6.97° |
| w384-s72-large | 96,000 / 32,000 | 512/512 | 115.0 | 18.52 mm / 5.90° |

Winner: **w384-s72-large, 512/512**. Faster narrow candidates remained slightly
less successful on this validation set. This is a finite comparison, not a claim
of globally optimal model quality. Raw strict validation acceptance was **1/2,048**.
The 8-step CPU smoke checkpoint remains separate and solved 0/32 raw tasks.

## Held-out IK and planning

40 tasks, seed 20261002: 24 independent FK pairs; eight tasks in two families
with identical XYZ and different directed tool Z from different currents; eight
zero/rest-start cases. Same-XYZ witness positions agree within 0.1 mm.
Success requires finite/joint-limit/self-collision validity and both 1 mm / 1°
goal tolerances. Numerical multiseed and NN-refined runs use the same default
four-goal budget; numerical-current intentionally uses one current seed/goal.

| Method | Valid IK / planned | Mean position / direction error | Inference ms | Refinement ms | Total IK ms | Planning ms |
|---|---:|---|---:|---:|---:|---:|
| numerical_current | 34/40 / 34/40 | 0.927 mm / 0.308° | 0.000 | 39.5 | 41.3 | 27.2 |
| numerical_multiseed | 40/40 / 40/40 | 0.929 mm / 0.299° | 0.000 | 490.2 | 498.6 | 44.3 |
| seed5_raw | 0/40 / 0/40 | — | 0.173 | 0.0 | 5.0 | 0.0 |
| seed5_refined | 40/40 / 40/40 | 0.928 mm / 0.277° | 0.161 | 334.3 | 345.7 | 43.4 |

Errors average IK-successful tasks only; individual failures/errors are retained
in `artifacts/seed5_v1/benchmark.json`. The final benchmark ran after training and
browser checks completed. Timings measure CPU computations, not motion time.
Every NN-refined task obtained at least one NN-refined accepted goal. Conventional
fallback supplied additional goals on some tasks to fill the default budget.
Raw NN proposals passed **0/40** tight held-out tasks; numerical refinement is
essential. NN-assisted IK was faster on this sample than equal-budget numerical
multiseed IK; current-only IK was faster but failed six tasks.
All 40 successful routes were direct connections, so this sample does not measure
PRM routing under blocked direct edges. Existing PRM/edge/failure tests cover those
behaviors. The reusable roadmap was retained and revalidated; it was not rebuilt.
Cold roadmap validation took 31.0 s and is separate from per-query timings.


## Verification

- 76 simulation tests passed, including 12 new Seed5 checks.
- 42 bridge/hardware tests passed using fake robots; no physical execution.
- 18 relevant browser checks passed across pose5, mouse-aim, PRM, vector entry,
  bounded DLS, target frames and workspace. The final trained-checkpoint vector
  preview was rerun successfully. Obsolete numerical-only defaults and the old
  workspace test port were corrected; cold graph validation is prewarmed in tests.
- Production WebUI build passed. Existing large-bundle advisory remains.
- Safe PyTorch checkpoint loading and NumPy parity passed: maximum proposal
  difference 7.42e-7 rad in the checked example.
- SHA256 matches the pre-upgrade snapshot for original `models.json`, `seed.pt`,
  `training.json`, `dataset.npz`, and `roadmap.json`.
- HTTP simulation dry run succeeded with both tolerances, four accepted goals,
  NN-refined/current sources, and a valid direct preview. The bridge remains
  hardware-disabled and disconnected. Artifact: `artifacts/seed5_v1/dry_run.json`.


## Changed files and artifacts

- Model/data/training: `seed5.py`, `seed5_data.py`, `train_seed5.py`,
  `select_seed5.py`, `benchmark_seed5.py`, `requirements-training.txt`;
  shared `kinematics.py` differentiable full FK and `train.py` optional hidden width.
- Solver/planning/API: `pose5.py`, `planning.py`, `service.py`, `workspace.py`.
- WebUI: `src/tool-direction.js`, `src/simulation-controls.js`,
  `src/workspace-controls.js`, `src/hardware-ik-controls.js`, `src/mouse-aim.js`,
  `src/main.js`, `src/style.css`, `index.html`, rebuilt `dist/`.
- Checks: `tests/test_seed5.py`, `../webui/tests/seed5.spec.js`,
  `../webui/tests/prm.spec.js`, `../webui/tests/workspace.spec.js`.
- Docs: `SEED5.md`, this file, `README.md`, `POSE5.md`, `PRM.md`,
  `SELF_COLLISION.md`, `../webui/README.md`.
- New versioned weights, datasets, smoke/candidate runs, training histories,
  validation selection, benchmark and dry run: `artifacts/seed5_v1/`.
- Corrected workflow: `../output/workflow/mycobot_seed5_ik_prm.{mmd,dot,svg,png}`.
- UI previews: `../webui/preview/seed5-vector-{desktop,mobile}.png`.


## Reproduction and remaining limits

See [SEED5.md](SEED5.md) for exact data-generation, GPU-training, candidate-selection,
smoke, benchmark and test commands, normalization/frames and actual source positions.
Use the existing simulation-only `./start_webui.sh` and reload the UI to use the
5D checkpoint. XYZ mode and its checkpoints remain available.

A seed NN does not replace goal validation or numerical IK. Finite sampling and
local optimization can fail on other goals; success here does not guarantee a route.
Intermediate orientation remains free. Collision checks use conservative URDF
self geometry and omit environment obstacles, attached tools/cables and calibration.
No physical robot motion or hardware performance measurement was performed.
