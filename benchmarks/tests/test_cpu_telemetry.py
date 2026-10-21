import copy
import unittest
from benchmarks.suite.cpu_telemetry import parse_process_stat, parse_host_stat, summarize_cpu


def sample(t, ticks, idle):
    return {'pid': 123, 'monotonic_s': t, 'available': True,
            'process': {'cpu_ticks': ticks, 'start_ticks': 99, 'last_main_thread_cpu': 2},
            'clock_ticks_per_second': 100, 'affinity': [2],
            'host_counters': {'cpu': {'total': int(t*100), 'idle': idle}},
            'main_thread_context_switches': {'voluntary_ctxt_switches': int(t*10)},
            'frequencies_khz': {'2': int(t*1000)}, 'governors': {'2': 'powersave'}}


class CpuTelemetryTests(unittest.TestCase):
    def test_process_stat_comm_parentheses_and_field_mapping(self):
        fields = ['0']*37
        fields[0] = 'R'; fields[11] = '11'; fields[12] = '5'
        fields[17] = '4'; fields[19] = '999'; fields[36] = '8'
        self.assertEqual(parse_process_stat('123 (name (x) y) '+' '.join(fields)),
                         {'cpu_ticks':16,'start_ticks':999,'threads':4,'last_main_thread_cpu':8})

    def test_guest_ticks_not_double_counted(self):
        self.assertEqual(parse_host_stat('cpu 10 2 3 40 5 1 1 0 7 1\ncpu0 1 0 1 8 0 0 0 0'),
                         {'cpu':{'total':62,'idle':45},'cpu0':{'total':10,'idle':8}})

    def test_rates_and_unknown_scheduler_wait(self):
        v = summarize_cpu([sample(1,100,40),sample(3,500,140)],0,4,123)
        self.assertTrue(v['available']); self.assertEqual(v['process_cpu_percent'],200)
        self.assertEqual(v['host_busy_percent']['cpu'],50)
        self.assertEqual(v['main_thread_context_switches']['voluntary_ctxt_switches'],20)
        self.assertIsNone(v['main_thread_runnable_wait_seconds'])
        self.assertEqual(v['frequency_khz_ranges']['2'],{'min':1000,'max':3000})

    def test_missing_failed_changed_identity_and_reset_do_not_claim_rates(self):
        a,b=sample(1,100,40),sample(3,500,140)
        for records in ([],[a],[a,{**b,'available':False}],
                        [a,{**b,'process':{**b['process'],'start_ticks':100}}],
                        [a,{**b,'process':{**b['process'],'cpu_ticks':0}}]):
            v=summarize_cpu(records,0,4,123)
            self.assertFalse(v['available']); self.assertIsNone(v['process_cpu_percent'])

    def test_only_window_and_pid_observations_are_used(self):
        a,b=sample(1,100,40),sample(3,500,140)
        foreign=sample(2,200,80); foreign['pid']=456
        v=summarize_cpu([sample(-2,0,0),a,foreign,b,sample(6,900,400)],0,4,123)
        self.assertEqual(v['samples'],2)


if __name__=='__main__': unittest.main()
