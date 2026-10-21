"""Timing diagnostics must preserve backend behavior and bounded samples."""
from concurrent.futures import ThreadPoolExecutor
import pytest
from object_tracking_ros2.timing import TimingStats, TimedBackend


def test_bounded_concurrent_samples_and_invalid_clock_values():
    stats = TimingStats(window=10)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: stats.add('stage', 2.), range(100)))
    for invalid in (-1., float('nan'), float('inf')):
        stats.add('stage', invalid)
    row = stats.snapshot()['stages']['stage']
    assert row == {'count': 100, 'window_count': 10, 'last_ms': 2.,
                   'median_ms': 2., 'p95_ms': 2., 'max_ms': 2.}


def test_backend_preserves_result_exception_and_attributes():
    response = {'result': {'server_compute_ms': 12., 'ik_elapsed_ms': 4.,
                'hardware_prepare_timing_ms': {'live_feedback_read': 100.,
                    'planner': 12., 'validation_and_encoder_preparation': 200., 'total': 312.}},
                'ik': {'telemetry_cycle_s': .2}}
    failure = RuntimeError('backend failed')
    class Backend:
        timeout = 3.
        def call(self, route, body=None):
            if route == '/api/fail':
                raise failure
            return response
    stats = TimingStats()
    wrapped = TimedBackend(Backend(), stats)
    assert wrapped.timeout == 3.
    assert wrapped.call('/api/plan', {}) is response
    with pytest.raises(RuntimeError) as caught:
        wrapped.call('/api/fail')
    assert caught.value is failure
    rows = stats.snapshot()['stages']
    assert rows['backend.fail']['count'] == 1
    assert rows['planner.server_compute']['last_ms'] == 12.
    assert rows['hardware_prepare.live_feedback_read']['last_ms'] == 100.
    assert rows['hardware_prepare.validation_and_encoder_preparation']['last_ms'] == 200.
    assert rows['hardware_prepare.total']['last_ms'] == 312.
    stats.backend({'feedback_pose': None})
    assert stats.snapshot()['backend_latest_seconds'] == {'telemetry_cycle_s': .2}
    assert 'telemetry_cycle_s' not in rows
