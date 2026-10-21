# Experimental one-pass trajectory transformer

This offline candidate tests a faster trajectory proposal model against the
active `diffusion_motion_v2` planner. It is not connected to the active WebUI
manifest and does not replace the hardware adapter.

## Model and training

`trajectory_transformer.py` implements a small conditional-VAE transformer:
192,486 parameters; width 64; four attention heads; a one-layer posterior encoder,
a two-layer trajectory decoder, and a 16-dimensional latent. Inputs use the same
34-feature motion-v2 schema (current/goal joints, one optional box, velocities,
accelerations and target velocity). Outputs are sixteen absolute bounded joint
positions, expressed as a learned residual to current normalized joints.

Inference has one decoder pass. Four proposals use one zero-latent candidate and
three sampled Gaussian latents. This is an ACT-inspired state-conditioned CVAE
experiment, not a reproduction of image-conditioned ACT or a diffusion model.
Every proposal still requires the existing collision and checked C2 controller
gates. The learned network is not a direct motor command or a collision guarantee.

The GPU run used the RTX 4070 Laptop GPU, batch 256, and 12,000 AdamW steps in
85.08 seconds. The saved EMA checkpoint comes from validation-selected step
11,000. The criterion updates the best score only for an improvement greater
than 1e-7; the step-12,000 metric differs by less than that threshold. The objective
is action MSE + 0.05 first-difference MSE + 0.0005 posterior KL. A fixed-seed,
best-of-four prior validation action MSE selects the checkpoint.

Sampling is 40% original expert episodes, 40% existing motion augmentation, and
20% new collision-protected episodes, then uniform frames within each episode.
Training retains the immutable motion-v2 cache and loads only train/validation
arrays. It does not load test arrays or hard-negative labels. Sources and source
hashes are recorded in `simulation/artifacts/trajectory_transformer_v1/model/training.json`.

## Comparison

`benchmark_trajectory_transformer.py` compares frozen weights with the pinned
motion-v2 SHA `6582cbf4e1d2575007cfead8f99ed5f15a7b7021083d0eb200c61b224401bddc`.
Both receive four proposals and the same checked C2 controller, target capture,
reference bounds and held-out tasks:

- 64 original fixed tasks; 200 decisions maximum.
- 24 moving/retarget tasks; 120 decisions maximum.
- 20 new collision-focused test tasks; 200 decisions maximum.
- Eight full service plans including shared learned IK, whole-path validation,
  smoothing, and independent complete-route replay at 0.01 radians.

Inference latency is measured on the GPU with synchronization, warmup, alternating
method order and three blocks of twenty pairs. Motion-v2 uses 32 DDPM steps and
its production CUDA-graph option; the candidate uses one decoder pass. This timing
includes output conversion but not IK, path checks, or hardware transport. The
virtual suites exclude computation delay from tracking RMSE; equal-duration goal
holds are included. Architecture, training budget and training data differ, so
this is a candidate-system comparison, not an isolated architecture experiment.

The candidate is not selected or tuned on those held-out scores. The historical
comparison corpus was previously used to evaluate motion-v2, and its baseline
results were known. No physical motion is sent by the benchmark. Actual results
are in `output/trajectory_transformer_comparison_20261008.json` and per-suite JSON.

## Commands

Use a fresh model directory for any new training run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  .venv/bin/python -m simulation.diffusion.train_trajectory_transformer \
  --output simulation/artifacts/trajectory_transformer_v1/new_model

MYCOBOT_IK_SEED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m simulation.diffusion.benchmark_trajectory_transformer \
  --output output/new_transformer_comparison.json
```

The new generator/model tests are in `test_collision_protected.py` and
`test_trajectory_transformer.py`. Existing diffusion/IK model artifacts,
launchers, hardware tolerances and runtime controllers remain the active system.

## Incoming-motion benchmark caveat

In the Oct 8 rerun, both policies rejected 17/24 historical initial braking
continuations before policy evaluation. A separate label audit found that 16
incoming-motion labels exceed the current conservative motion envelope. This
is a shared controller/data compatibility issue; the 7/24 score must not be
interpreted as 17 neural policy failures.

The benchmark appends a diagnostic using the same 24 held-out target sequences,
starting at the same joint positions with zero initial velocity/acceleration.
It retains every controller gate. The diagnostic is explicitly labeled and
neither model is retrained or selected using those scores. To append only that
diagnostic to an existing frozen comparison, use `--moving-from-rest-only`.

## Frozen Oct 8 results

| Measurement | Active motion-v2 | Transformer |
|---|---:|---:|
| Original fixed tasks | 62/64 | 50/64 |
| Original block-scene subset | 30/32 | 18/32 |
| Original no-block subset | 32/32 | 32/32 |
| New collision-focused tasks | 17/20 | 14/20 |
| Moving targets from rest (diagnostic) | 22/24 | 21/24 |
| Complete IK-to-timed plans | 8/8 | 5/8 |
| GPU inference median, four proposals | 16.05 ms | 0.88 ms |
| GPU inference p95 | 16.78 ms | 1.05 ms |

Both policies had zero collisions in checked virtual execution. The transformer
produced more rejected raw paths and failed the success, full-plan, moving-from-rest
and tracking-error replacement gates. Faster proposal inference does not establish
faster successful complete planning. Keep motion-v2 selected. Neither the active
manifest, hardware controller nor existing model artifacts were changed.

The result PDF is `output/reports/transformer_comparison_20261008.pdf`. Six focused
checks passed; their log is `output/collision_transformer_tests_20261008.log`.
