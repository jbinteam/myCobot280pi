"""Read-only Linux CPU observations from the benchmark parent process.

Host busy time includes this benchmark. Sampled frequencies and the main
thread's last CPU cannot establish frequency throttling or CPU migration history.
"""
import os
from pathlib import Path
import time


def _text(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeError):
        return None


def parse_process_stat(text):
    # comm may contain spaces and parentheses; fields after its final ')' are fixed.
    fields = text[text.rfind(')')+2:].split()
    if len(fields) < 37:
        raise ValueError('Incomplete process stat')
    return {'cpu_ticks': int(fields[11])+int(fields[12]),
            'start_ticks': int(fields[19]), 'threads': int(fields[17]),
            'last_main_thread_cpu': int(fields[36])}


def parse_host_stat(text):
    result = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields or not fields[0].startswith('cpu'):
            continue
        if fields[0] != 'cpu' and not fields[0][3:].isdigit():
            continue
        # Linux guest time is included in user/nice already: never double count it.
        values = [int(v) for v in fields[1:9]]
        if len(values) < 5:
            raise ValueError('Incomplete CPU counters')
        result[fields[0]] = {'total': sum(values), 'idle': values[3]+values[4]}
    return result


def sample_cpu(pid):
    began = time.monotonic()
    value = {'pid': pid, 'query_started_monotonic_s': began, 'available': False}
    try:
        value['process'] = parse_process_stat(Path(f'/proc/{pid}/stat').read_text())
        value['clock_ticks_per_second'] = os.sysconf('SC_CLK_TCK')
        value['host_counters'] = parse_host_stat(Path('/proc/stat').read_text())
        value['affinity'] = sorted(os.sched_getaffinity(pid))
        status = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in line)
        value['main_thread_context_switches'] = {
            name: int(status[name]) for name in ('voluntary_ctxt_switches', 'nonvoluntary_ctxt_switches') if name in status}
        value['load_average'] = list(os.getloadavg())
        value['frequencies_khz'] = {}
        value['governors'] = {}
        for cpu in value['affinity']:
            folder = Path(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq')
            frequency = _text(folder/'scaling_cur_freq')
            if frequency and frequency.isdigit():
                value['frequencies_khz'][str(cpu)] = int(frequency)
            governor = _text(folder/'scaling_governor')
            if governor:
                value['governors'][str(cpu)] = governor
        value['scheduler_stats_enabled'] = _text('/proc/sys/kernel/sched_schedstats') == '1'
        sched = _text(f'/proc/{pid}/schedstat')
        if value['scheduler_stats_enabled'] and sched:
            fields = sched.split()
            value['main_thread_scheduler'] = {'runtime_ns': int(fields[0]), 'runnable_wait_ns': int(fields[1]), 'timeslices': int(fields[2])}
        value['available'] = True
    except (OSError, ValueError, IndexError) as exc:
        value['error'] = f'{type(exc).__name__}: {exc}'
    value['query_finished_monotonic_s'] = time.monotonic()
    value['monotonic_s'] = (began+value['query_finished_monotonic_s'])/2
    return value


def summarize_cpu(samples, started, finished, pid):
    selected = [s for s in samples if s.get('pid') == pid and
                s.get('query_started_monotonic_s', s['monotonic_s']) <= finished and
                s.get('query_finished_monotonic_s', s['monotonic_s']) >= started]
    valid = sorted((s for s in selected if s.get('available')), key=lambda s: s['monotonic_s'])
    interior = [s for s in valid if started <= s['monotonic_s'] <= finished]
    first, last = (interior[0], interior[-1]) if interior else (None, None)
    span = last['monotonic_s']-first['monotonic_s'] if first else 0.
    identity = {s['process']['start_ticks'] for s in valid}
    available = len(interior) >= 2 and span > 0 and len(identity) == 1 and all(s.get('available') for s in selected)
    result = {'available': available, 'samples': len(selected), 'valid_samples': len(valid),
              'window_start_monotonic_s': started, 'window_end_monotonic_s': finished,
              'observed_span_s': span, 'process_cpu_seconds': None, 'process_cpu_percent': None,
              'host_busy_percent': {}, 'main_thread_context_switches': {},
              'main_thread_runnable_wait_seconds': None,
              'affinity_observed': sorted({cpu for s in valid for cpu in s.get('affinity', [])}),
              'last_main_thread_cpus_observed': sorted({s['process']['last_main_thread_cpu'] for s in valid}),
              'frequency_khz_ranges': {}, 'governors_observed': sorted({g for s in valid for g in s.get('governors', {}).values()}),
              'limitations': 'CPU rates use first/last interior samples, not exact plan boundaries. Process CPU sums all process threads and may exceed 100% of one core. Host busy time includes the benchmark; it does not identify competitors. Context switches and scheduler wait describe the main thread only. Frequency snapshots across allowed CPUs do not prove throttling, actual execution frequency, or migration history.'}
    times = [started, *(s['monotonic_s'] for s in interior), finished]
    result['maximum_sample_gap_s'] = max((b-a for a,b in zip(times,times[1:])), default=finished-started)
    for cpu in result['affinity_observed']:
        frequencies = [s.get('frequencies_khz', {}).get(str(cpu)) for s in valid]
        frequencies = [v for v in frequencies if v is not None]
        if frequencies:
            result['frequency_khz_ranges'][str(cpu)] = {'min': min(frequencies), 'max': max(frequencies)}
    if available:
        ticks = last['process']['cpu_ticks']-first['process']['cpu_ticks']
        if ticks < 0 or last['clock_ticks_per_second'] != first['clock_ticks_per_second']:
            result['available'] = False
        else:
            result['process_cpu_seconds'] = ticks/first['clock_ticks_per_second']
            result['process_cpu_percent'] = 100*result['process_cpu_seconds']/span
        for cpu, end in last['host_counters'].items():
            begin = first['host_counters'].get(cpu)
            if begin is None: continue
            total, idle = end['total']-begin['total'], end['idle']-begin['idle']
            # Hotplug/reset/iowait accounting can make deltas unusable; do not clamp.
            if total > 0 and 0 <= idle <= total:
                result['host_busy_percent'][cpu] = 100*(total-idle)/total
        for name, end in last.get('main_thread_context_switches', {}).items():
            begin = first.get('main_thread_context_switches', {}).get(name)
            if begin is not None and end >= begin:
                result['main_thread_context_switches'][name] = end-begin
        a, b = first.get('main_thread_scheduler'), last.get('main_thread_scheduler')
        if a and b and all(s.get('scheduler_stats_enabled') for s in interior):
            wait = b['runnable_wait_ns']-a['runnable_wait_ns']
            if wait >= 0: result['main_thread_runnable_wait_seconds'] = wait/1e9
    if not result['available']:
        result['unavailable_reason'] = 'Insufficient valid interior observations, process identity changed, or failed CPU observations.'
    return result
