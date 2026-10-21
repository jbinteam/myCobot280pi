"""Bounded, thread-safe diagnostic timing; never changes control decisions."""
from collections import deque
from functools import wraps
import math
from statistics import median
import threading
from time import perf_counter


class TimingStats:
    def __init__(self, window=512):
        self.window = window
        self.lock = threading.Lock()
        self.samples = {}
        self.counts = {}
        self.backend_latest = {}

    def add(self, name, milliseconds):
        if not math.isfinite(milliseconds) or milliseconds < 0:
            return
        with self.lock:
            self.samples.setdefault(name, deque(maxlen=self.window)).append(float(milliseconds))
            self.counts[name] = self.counts.get(name, 0)+1

    def backend(self, result):
        planned = result.get('result') or {}
        for key in ('server_compute_ms', 'ik_elapsed_ms', 'path_elapsed_ms'):
            value = planned.get(key)
            if isinstance(value, (int, float)):
                self.add('planner.'+key.removesuffix('_ms'), value)
        preparation = planned.get('hardware_prepare_timing_ms') or {}
        for key in ('live_feedback_read', 'planner', 'validation_and_encoder_preparation', 'total'):
            value = preparation.get(key)
            if isinstance(value, (int, float)):
                self.add('hardware_prepare.'+key, value)
        ik = result.get('ik') or {}
        keys = ('telemetry_cycle_s', 'actual_send_interval_s', 'deadline_wait_s',
                'send_deadline_lateness_s', 'reference_lag_s')
        with self.lock:
            # These are backend snapshots, not independent hardware samples.
            # Do not aggregate repeated polls into misleading percentiles.
            if 'ik' in result:
                self.backend_latest = {k: ik[k] for k in keys if k in ik}

    def snapshot(self):
        with self.lock:
            samples = {k: (list(v), self.counts[k]) for k, v in self.samples.items()}
            latest = dict(self.backend_latest)
        stages = {}
        for name, (values, count) in samples.items():
            ordered = sorted(values)
            stages[name] = {'count': count, 'window_count': len(values),
                'last_ms': values[-1], 'median_ms': median(values),
                'p95_ms': ordered[math.ceil(.95*len(ordered))-1], 'max_ms': ordered[-1]}
        return {'window_size': self.window, 'stages': stages,
                'backend_latest_seconds': latest}


def timed(name):
    """Measure a node callback, including its waits; timings may be nested."""
    def decorate(function):
        @wraps(function)
        def wrapped(self, *args, **kwargs):
            began = perf_counter()
            try:
                return function(self, *args, **kwargs)
            finally:
                self.timings.add(name, (perf_counter()-began)*1000)
        return wrapped
    return decorate


class TimedBackend:
    def __init__(self, backend, timings):
        self.backend = backend
        self.timings = timings

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def call(self, route, body=None):
        began = perf_counter()
        try:
            result = self.backend.call(route, body)
            self.timings.backend(result)
            return result
        finally:
            self.timings.add('backend.'+route.removeprefix('/api/').replace('/', '.'),
                             (perf_counter()-began)*1000)

    def plan_with_observer(self, body, observer):
        began = perf_counter()
        try:
            if callable(getattr(self.backend, 'plan_with_observer', None)):
                result = self.backend.plan_with_observer(body, observer)
            else:
                result = self.backend.call('/api/ik/mouse-aim/plan', body)
            self.timings.backend(result)
            return result
        finally:
            self.timings.add('backend.ik.mouse-aim.plan', (perf_counter()-began)*1000)
