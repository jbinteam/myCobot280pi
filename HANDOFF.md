# Project handoff

Saved: 2026-10-21T19:24:52+07:00 (Asia/Bangkok).
Project: myCobot280pi. Root initialized as a Git repository on main for the authorized initial publication.
Objective: transfer this research/control project to a junior engineer; owner will push to GitHub.

## Completed this session

- Added root README.md and docs/HANDOVER.md: architecture, code map, setup,
  first-day hands-on simulation exercise, constraints, troubleshooting and backlog.
- Added docs/GITHUB.md and root .gitignore, separating runtime source/models
  from multi-GB datasets, environments, outputs and nested upstream checkouts.
- Added requirements-runtime.txt and docs/environment-observed.txt.
  The latter is an observed environment snapshot, not a portable lockfile.
- Added docs/artifact-manifest.json and scripts/verify_handover_artifacts.py
  for 102 retained artifact/geometry files.
- Copied the historical October 20 validation summary to
  docs/evidence/ros2-camera-fallback-20261020.json so it survives Git exclusions.
- No controller behavior changed; no physical commands, connections, remote
  deployments, Git initialization, commits or pushes were performed.

## Current baseline and historical evidence

- Active fixed-target/tracking planner: learned IK + classical RRT*.
  Diffusion v2 is retained. Both pinned model readiness checks pass.
- Default ROS frontend is C++ with embedded shared Python controller;
  the Python ROS frontend remains available.
- Current ROS source/docs include October 20 strict-first camera-FOV fallback,
  measured-camera visualization and cooperative Stop/loss/disconnect cancellation.
- The retained October 20 summary reports 141 ROS, 1 native, 88 shared,
  47 planner/cancellation and 2 focused resume checks passed in separate runs.
  These historical counts may overlap and were not rerun in this session.
  They used fake motors/synthetic detections, not physical validation.
- Runtime: Python 3.12.3, ROS Jazzy, Node 18.19.1; SDK checkout clean at
  44d7fb1c34e1d6c3924ec86617e8f9637f449074. Root has no Git history.
- Prior fixed-target public illustration and hosting history remain in the
  archived handoff below. Local source: output/anatomy_arm_approach_20261015/.
  Hosting belongs to separate nn_ik/webui project; no remote state checked now.
  Retain user's fixed-target preference; moving-target variant was rolled back.

## Verification actually run now

- npm run build --prefix webui: passed (51 modules); existing large-chunk warning.
- OPENBLAS_NUM_THREADS=1 PYTHONPATH="$PWD/object_tracking_ros2:$PWD"
  python3 -m pytest -q object_tracking_ros2/test/test_core.py
  object_tracking_ros2/test/test_runtime_options.py: 62 passed in 0.15s.
- Pinned diffusion verified_status and classical_service.info: both ready.
  Diffusion hash: 6582cbf4e1d2575007cfead8f99ed5f15a7b7021083d0eb200c61b224401bddc.
  Learned IK hash: 168ab43a2d4d4dcc0e61acf8c3c89e6dff6e9723843b1fd023a4519108ea04f5.
- python3 scripts/verify_handover_artifacts.py: all 102 files verified.
- Used a temporary external Git directory to inspect proposed inclusion rules;
  project Git state was not changed. No retained file exceeded 13 MB.
- No clean-machine installation, full ROS rebuild/test suite, browser interaction,
  hardware trial, detector calibration or live deployment test performed now.

## Constraints and open work

- Camera mount/FOV remain assumed and uncalibrated; physical dynamics, Stop
  latency and full-object visibility are not qualified by simulation evidence.
- Atom 6.5 directional jog remains blocked after the October 6 overtravel;
  independent Pi-side watchdog/firmware validation remain outstanding.
- Preserve measured starts, frozen stage retries, arrival-gated advancement,
  full route/collision/visibility/timing/rounded-encoder validation and cancellation.
- Separate objectScanner source/dataset/camera runtime and Pi vendor Server_280.py
  must be transferred for the full lab. Training history and output archives are
  excluded from Git; preserve them separately if the junior needs research history.
- First concrete next step: owner reviews docs/GITHUB.md and proposed staged
  files, publishes to the chosen repository, then junior completes the clean-clone
  simulation exercise in docs/HANDOVER.md before supervised hardware work.
- No remaining documentation blocker. Fresh-clone reproducibility and physical
  commissioning are explicitly unfinished engineering acceptance tasks.

Previous handoff archived at: .codex/handoffs/2026-10-21T19-24-52+0700-b6a61768.md.
AGENTS.md already contains startup continuity and /handoff behavior; unchanged.

## GitHub publication follow-up

- Owner authorized pushing to https://github.com/jbinteam/myCobot280pi.git.
- Remote advertised no refs before initialization; origin now points there.
- Initial commit includes source, runtime artifacts and handover documentation.
- Staged-file checks exclude local environments, outputs and nested checkouts;
  targeted credential-pattern scan found no matches (not a comprehensive audit).
- Initial whitespace check reports pre-existing whitespace in research/tests and
  vendored geometry. Geometry is preserved byte-for-byte for pinned hashes.
- Use `git status`, `git log -1` and `git ls-remote origin refs/heads/main` to
  confirm current publication state; earlier non-Git notes describe prior state.
