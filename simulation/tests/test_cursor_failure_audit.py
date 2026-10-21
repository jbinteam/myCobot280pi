"""Guard audit conclusions against actual FK witnesses, including tolerance edges."""
import json
from pathlib import Path
import unittest

import numpy as np

from simulation import pose5
from simulation.audit_cursor_failures import cursor_metrics, endpoint
from simulation.cursor_geometry_audit import geometry_certificate, model_bounds
from simulation.kinematics import ARM, rotation
from simulation.prm import JointSpace


class CursorGeometryAuditTest(unittest.TestCase):
    def test_uniform_rounding_bounds_cover_exact_urdf(self):
        bounds = model_bounds()
        rng = np.random.default_rng(280)
        for q in rng.uniform(ARM.lower, ARM.upper, (500, 6)):
            ideal = np.eye(4)
            joint = 0
            for base, axis in ARM.chain:
                snapped = base.copy()
                snapped[:3, :3] = np.round(base[:3, :3])
                ideal = ideal@snapped
                if axis is not None:
                    r = np.eye(4)
                    r[:3, :3] = rotation(axis, q[joint])
                    ideal = ideal@r
                    joint += 1
            exact = ARM.transform(q)
            self.assertLessEqual(np.linalg.norm(exact[:3, 3]-ideal[:3, 3]), bounds['position_m'])
            self.assertLessEqual(np.linalg.norm(exact[:3, 2]-ideal[:3, 2]), bounds['axis_chord'])

    def test_no_false_exclusions_for_full_range_fk_witnesses(self):
        rng = np.random.default_rng(20261002)
        configurations = list(rng.uniform(ARM.lower, ARM.upper, (1000, 6)))
        for joint in range(6):
            for bound in (ARM.lower, ARM.upper):
                q = ARM.center.copy()
                q[joint] = bound[joint]
                configurations.append(q)
        for q in configurations:
            t = ARM.transform(q)
            certificate = geometry_certificate(t[:3, 3], t[:3, 2], 0., 0.)
            self.assertFalse(certificate['geometric_infeasible_with_tolerances'], (q, certificate))

    def test_tolerance_edge_witnesses_are_not_excluded(self):
        rng = np.random.default_rng(5102)
        for index, q in enumerate(rng.uniform(ARM.lower, ARM.upper, (1000, 6))):
            if index % 6 == 0:
                joint = (index//6) % 6
                q[joint] = ARM.lower[joint] if index % 12 == 0 else ARM.upper[joint]
            t = ARM.transform(q)
            displacement = pose5.unit_axis(rng.normal(size=3))*.001
            turn_axis = np.cross(t[:3, 2], pose5.unit_axis(rng.normal(size=3)))
            axis = rotation(turn_axis, np.deg2rad(1.))@t[:3, 2]
            # This exact FK configuration meets both tolerances at their edges.
            certificate = geometry_certificate(t[:3, 3]+displacement, axis)
            self.assertFalse(certificate['geometric_infeasible_with_tolerances'], (q, certificate))

    def test_saved_controls_and_failure_counts(self):
        root = Path(__file__).resolve().parents[2]
        benchmark = json.loads((root/'output/ik_recovery/benchmark.json').read_text())['cursor']
        cases = {case['id']: case for case in benchmark['tasks']}
        excluded_failures = 0
        unresolved = []
        for saved in benchmark['rows']['upgraded']:
            case = cases[saved['task_id']]
            cursor = np.asarray(case['cursor_mm'])/1000
            axis = pose5.unit_axis(cursor-ARM.fk(np.deg2rad(case['start_deg'])))
            excluded = geometry_certificate(cursor-.05*axis, axis)['geometric_infeasible_with_tolerances']
            if saved['planned']:
                self.assertFalse(excluded, case['id'])
            elif excluded:
                excluded_failures += 1
            else:
                unresolved.append(case['id'])
        self.assertEqual(excluded_failures, 17)
        self.assertEqual(unresolved, ['start-0-cursor--60-120', 'start-0-cursor-120-120'])

    def test_endpoint_acceptance_keeps_nominal_and_cursor_constraints(self):
        q = np.deg2rad([0, -40, -130, 80, 0, 50])
        t = ARM.transform(q)
        cursor = (t[:3, 3]+.05*t[:3, 2])*1000
        self.assertTrue(cursor_metrics(q, cursor)['accepted'])
        spun = q.copy()
        spun[-1] += .3
        self.assertTrue(cursor_metrics(spun, cursor)['accepted'])
        evidence = endpoint(q, t[:3, 3]+[.002, 0, 0], t[:3, 2], cursor, JointSpace(), 'test')
        self.assertTrue(evidence['cursor']['accepted'])
        self.assertGreater(evidence['nominal_tolerance_ratio'], 1.)
        self.assertGreater(evidence['all_tolerance_ratio'], 1.)


if __name__ == '__main__':
    unittest.main()
