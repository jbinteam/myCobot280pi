# Preparing the GitHub transfer

The project root was a non-Git workspace when this handover was prepared.
The owner subsequently authorized publication to
https://github.com/jbinteam/myCobot280pi.git. The root is now initialized on
`main` with that repository as `origin`. The commands below document the initial
publishing procedure; do not repeat initialization or remote creation in a clone.

## Included source and runtime assets

The root `.gitignore` retains the active Python/C++/browser source, ROS packages,
launchers, tests, robot URDF/meshes, model weights and deployment manifests.
Runtime artifacts are small enough to remain in this proposed source set;
training datasets and caches are excluded. Check the complete proposed file
list before committing. The 102 retained artifact/geometry files have SHA-256
identities in `artifact-manifest.json`.

```bash
python3 scripts/verify_handover_artifacts.py
```

This checks transfer integrity, not controller behavior. Startup additionally
checks pinned checkpoint/schema/geometry compatibility.

## Excluded local material

| Excluded | Transfer/rebuild procedure |
| --- | --- |
| `.venv`, `node_modules`, browser/ROS builds, caches | Recreate using HANDOVER.md and launchers |
| Separate `pymycobot`, `catkin_ws`, `colcon_ws` checkouts | Install pinned SDK; active URDF is already vendored |
| Training HDF5, NumPy datasets, shards, caches, candidates, backups | Transfer original research archive separately if retraining is assigned |
| `output`, `tmp`, logs, `.codex` | Local history; selected validation summary retained in docs/evidence |
| AppImage and research paper downloads | Obtain separately; retain citations in research.md |
| Environment secrets and private key files | Configure per machine; never publish |

Exclusion does not delete any local file. Old component documents can link to
excluded research outputs; these links require the separately transferred archive.
The public fixed-target illustration source also lives in that archive. The
website project, objectScanner detector/model, and Pi vendor server are separate
handover items; this GitHub repository alone does not supply the full lab.

## Owner's publish procedure

From this project root, when ready:

```bash
git init -b main
git add --dry-run .
# Inspect the proposed paths, then stage and review:
git add .
git diff --cached --stat
git diff --cached --check
git status --short
# After reviewing content:
git commit -m "Prepare myCobot controller and junior engineer handover"
# Substitute your actual GitHub repository URL:
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

Before publishing, review staged configuration for machine-specific addresses,
local paths and credentials. Ignore patterns are not a secret scanner. Choose
repository visibility and licensing deliberately; upstream URDF/mesh licensing
and SDK provenance must remain intact. Package metadata contains MIT declarations,
but no new project-wide license grant was invented in this handover.

After pushing, use a fresh clone on the junior's machine. Run the artifact
verification, dependency installation, model readiness checks, WebUI build and
62-test quick suite documented in HANDOVER.md. Then start disconnected ROS.
Record failures and environment differences before any physical commissioning.
A fresh-clone installation remains an outstanding acceptance step.
