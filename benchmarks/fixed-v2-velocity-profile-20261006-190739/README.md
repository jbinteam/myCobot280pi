# Current fixed-v2 velocity-profile benchmark

Saved 2026-10-06T19:07:39.404279+07:00. This freezes the restored full-horizon C2 rollout, final collision-checked shortcuts, selected-speed quintic timing and encoder velocity-profile driver.

The saved reference has 622 frames over 12.42 seconds. All q(t), velocity and acceleration comparisons have zero difference. The longer completed-plan replay has 994 frames over 19.86 seconds. Defaults: seed 42, four candidates, 32 denoising steps, 200 decision limit, 5% joint speed allowance and a 0.2-second command period.

83 Python tests and two browser tests passed. The latest hardware execution stopped because its endpoint did not settle within ten seconds; its FK error was 9.025948659001864 mm. The earlier successful execution and this later failure are both retained in benchmark.json. Physical speed conversion remains nominal.

Backup: `../../backups/fixed-v2-velocity-profile-20261006-190739.tar.gz`. Extract into a separate directory to inspect the snapshot; do not overwrite a running controller. The archive contains files at their original project-relative paths and a SHA256SUMS file. It is an overlay for this project, not a bundled Python/Node environment.

To reproduce the archived profile from the project root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python tmp/validate_v2_rollback.py
```

This replay sends no robot commands. Compare live trials only when start joints, target, speed and sampling settings match.
