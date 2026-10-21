# Collision-protected demonstrations

`generate_collision_protected.py` creates a new, offline dataset compatible with
the existing 34-feature motion input and `train_motion` augmentation fields.
It does not change robot geometry, the active deployment, or prior datasets.

## Protection and coverage

Each of four families has equal representation:

- No block: retain self-collision, table and operating joint limits.
- Diverse box: table-mounted boxes with broader centre, size and height coverage.
- Blocked box: endpoints are valid but their direct connection is rejected;
  a checked RRT* route must detour.
- Near boundary: a valid endpoint is accompanied by a collision-invalid neighbour
  within 0.02 radians in joint space. This is not a metric clearance estimate.

RRT* and shortcuts use connecting-edge checks at 0.01 radians. Learning references
are analytic quintics on a 0.1-second grid. The exact same timed curves are replayed
at intervals no larger than 0.01 seconds (100 Hz); changing the replay grid does
not retime the route. Every dense pose and connecting edge must pass the same
self/table/box/limit checks with 1 mm model clearance. Analytic reference speed
and acceleration remain at most 1 rad/s and 3 rad/s² in vector norm.

After generation, a fresh scene independently reopens and checks every episode,
its timing/derivative labels, scene category, blockage label, and hard-negative
evidence. Publication requires that audit to pass. Failed final replay examples
are excluded. Existing outputs, partial files and shards are protected against
accidental overwrite.

## Actual generated artifacts

`simulation/artifacts/collision_protected_v1/training_1000.h5` has 1,000 episodes:
250 per family; 800 training, 100 validation and 100 test episodes. There are
61,208 learning states and 603,080 dense replay states. Its adjacent JSON records
checksums, sources, seeds, rejection counts and the independent replay result.
A separate 120-episode pilot is also preserved.

The root `episodes` contains positive action labels. The root `hard_negatives`
contains 250 rejected neighbouring poses, linked to positive episodes; it must
never be treated as a trajectory or appended to action windows. The current
transformer trainer uses positive episodes only. These negatives are evidence
and potential inputs for a separately designed future collision classifier.

Episode-level splits are stratified by family; exact duplicate tasks are rejected
by the audit. No original dataset or pinned model was overwritten. The schema
supports a single axis-aligned box. Multiple obstacles, tool geometry and physical
dynamics require additional modeling; this dataset is not a continuous-collision
certificate or physical qualification.

## Reproduce or audit

Choose a fresh output path to generate another set:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m simulation.diffusion.generate_collision_protected \
  --output simulation/artifacts/collision_protected_v1/new_set.h5 \
  --episodes 1000 --workers 4 --seed 202610082

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m simulation.diffusion.generate_collision_protected \
  --audit simulation/artifacts/collision_protected_v1/training_1000.h5
```

The generated motion fields can be supplied as `--augmentation` to `train_motion`
with fresh cache/model paths. That replaces its augmentation input; it does not
implicitly merge the existing motion augmentation. The new transformer trainer
explicitly retains the existing motion data and adds the protected positive data.
