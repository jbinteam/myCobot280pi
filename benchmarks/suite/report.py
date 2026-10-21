"""Self-contained escaped HTML; report data never executes as script."""
import html
import json
from pathlib import Path
from .metrics import distribution


def write_report(directory, report):
    directory = Path(directory)
    (directory/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    esc = lambda x: html.escape(str(x), quote=True)
    cell = lambda values: '<tr>'+''.join('<td>'+esc(v)+'</td>' for v in values)+'</tr>'
    fmt = lambda value: '—' if value is None else f'{value:.3f}'
    rows, details, timing_rows = [], [], []
    for group in report['summary']['groups']:
        def timing(backend):
            d = group['warm_success_latency_ms'][backend]
            return '—' if not d['n'] else f"{d['median']:.2f} [{d['min']:.2f}, {d['max']:.2f}], n={d['n']}"
        counts = group['status_counts']
        def outcomes(backend):
            return ', '.join(f'{status}: {count}' for status,count in counts[backend].items()) or 'missing'
        outcome = outcomes('eager') if counts['eager'] == counts['cuda_graph'] else 'eager '+outcomes('eager')+'; graph '+outcomes('cuda_graph')
        ratio = group['speedup_ratio']
        rows.append(cell((group['case_id'], group['seed'], group['expected']['kind'], outcome,
                         'PASS' if group['correctness_passed'] else 'FAIL', timing('eager'), timing('cuda_graph'),
                         '—' if ratio is None else f'{ratio:.3f}×', '; '.join(group['errors']))))
        paired = group.get('paired_timing', {})
        interval = paired.get('bootstrap_ci95')
        ci = '—' if not interval else f"[{interval['lower']:.3f}, {interval['upper']:.3f}]"
        def dispersion(backend):
            d = group['warm_success_latency_ms'][backend]
            return '—' if d.get('iqr') is None else f"{d['iqr']:.2f} / {d['mad']:.2f}"
        timing_rows.append(cell((group['case_id'], group['seed'], group.get('timing_status', 'descriptive'),
                                paired.get('complete_block_count', 0), fmt(paired.get('median_ratio')), ci,
                                dispersion('eager'), dispersion('cuda_graph'),
                                '; '.join(group.get('timing_reasons', [])))))
        detail_rows = []
        for backend in ('eager', 'cuda_graph'):
            records = [r for r in report.get('records', []) if r.get('case_id') == group['case_id'] and
                       r.get('seed') == group['seed'] and r.get('backend') == backend and not r.get('error')]
            def median_stage(key):
                values = [r['timing_ms'][key] for r in records if r.get('timing_ms', {}).get(key) is not None]
                return fmt(distribution(values)['median'])
            successful = [r for r in records if r.get('status') == 'success' and r.get('converged') is True]
            def maximum_metric(key):
                values = [r['metrics'][key] for r in successful if r.get('metrics', {}).get(key) is not None]
                return fmt(max(values)) if values else '—'
            detail_rows.append(cell((backend, len(records), *(median_stage(k) for k in
                ('inference','ik','collision','smoothing','offline_encoder_preparation')),
                *(maximum_metric(k) for k in ('endpoint_error_mm','endpoint_direction_error_deg',
                                             'duration_s','joint_route_length_rad','max_velocity_rad_s','max_acceleration_rad_s2')))))
        details.append('<details><summary>'+esc(group['case_id'])+' / seed '+esc(group['seed'])+
                       '</summary><div class="table-scroll"><table><thead><tr>'+''.join('<th>'+esc(v)+'</th>' for v in
                       ('Backend','Trials','Inference ms','IK ms','Collision ms','Smoothing ms','Encoder prep ms',
                        'FK error mm','FK axis error °','Duration s','Joint route rad','Sampled speed rad/s','Sampled acceleration rad/s²'))+
                       '</tr></thead><tbody>'+''.join(detail_rows)+'</tbody></table></div></details>')
    payload = esc(json.dumps(report, indent=2, allow_nan=False))
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Offline planner benchmark</title><style>body{font:15px system-ui;margin:2rem;color:#18222c}table{border-collapse:collapse;width:100%}th,td{padding:.6rem;border:1px solid #ccd3da;text-align:left}th{background:#edf2f6}pre{white-space:pre-wrap;overflow-wrap:anywhere}p{max-width:90ch}.table-scroll{overflow-x:auto;margin-bottom:1rem}details{margin:1rem 0}summary{cursor:pointer}</style>
<h1>Offline planner benchmark: STATUS</h1><p>Current-source pinned v2 eager versus current-source CUDA graph inference. This is not a replay of the October 6 frozen implementation.</p>
<p>Hardware qualification: UNTESTED. Position errors are URDF forward-kinematics estimates, not external measurements. No physical IO is permitted.</p>
<p>Warm latency is instrumented planning time, shown as median [min, max] in milliseconds. Only valid successful trials contribute. Cold process/model setup and case warmups are separate in the data. Hashing overhead is included equally; timings are not UI or transport latency. p95 is withheld below 20 samples. Failed/incomplete comparisons have no speedup claim.</p>
<div class="table-scroll"><table><thead><tr><th>Case</th><th>Seed</th><th>Expectation</th><th>Actual outcome / backend</th><th>Correctness</th><th>Eager ms</th><th>Graph ms</th><th>Speedup</th><th>Errors</th></tr></thead><tbody>ROWS</tbody></table></div>
<h2>Timing confidence</h2><p>Correctness PASS and timing support are separate. Quick/full comparisons are descriptive. Timing mode requires at least 10 complete ABBA blocks, successful identical plans, usable GPU telemetry without observed foreign compute processes, and acceptable variation before reporting a supported speedup. Inconclusive results retain all raw samples.</p>
<p>Each block ratio is mean eager latency divided by mean graph latency. The table shows the median block ratio and an approximate 95% percentile bootstrap interval from 10,000 resamples of whole blocks. These are per-case exploratory intervals, not simultaneous guarantees across cases. Blocks can still share time-dependent system effects; fresh processes do not guarantee independence. IQR/median above 10% or a first-half versus second-half block-ratio change above 10% withholds support. These are declared diagnostic thresholds, not proofs of stability.</p>
<p>Dispersion columns show IQR / median absolute deviation in milliseconds. Quartiles use linear interpolation; MAD is unscaled. GPU sampling covers warmup and measurement windows, approximately once per second, and can miss shorter activity. Clock and temperature ranges are diagnostics, not corrections. Instrumentation and monitoring have overhead; these results describe this offline workload on this machine.</p>
<div class="table-scroll"><table><thead><tr><th>Case</th><th>Seed</th><th>Timing status</th><th>ABBA blocks</th><th>Median block ratio</th><th>Approx. 95% CI</th><th>Eager IQR / MAD ms</th><th>Graph IQR / MAD ms</th><th>Reasons</th></tr></thead><tbody>TIMINGROWS</tbody></table></div>
<h2>Stage timing and reference quality</h2><p>Stages show medians of completed instrumented trials. Nested stages are exclusive; collision excludes scalar checks outside the measured collision functions. Encoder preparation is separate from planning total. Reference quality columns show the maximum observed value among successful trials, including sampled finite-difference speed/acceleration maxima; these are not analytic derivative bounds or physical measurements. Exploratory cases are marked “allowed”; convergence is unqualified. Failed comparisons remain diagnostic only.</p>DETAILS
<h2>Failures</h2><pre>FAILURES</pre><details><summary>Complete reproducibility record</summary><pre>PAYLOAD</pre></details></html>'''
    page = page.replace('STATUS', 'PASS' if report['summary']['passed'] else 'FAIL').replace('TIMINGROWS', ''.join(timing_rows)).replace('ROWS', ''.join(rows)).replace('DETAILS', ''.join(details)).replace('FAILURES', esc('\n'.join(report['summary']['failures']) or 'None')).replace('PAYLOAD', payload)
    (directory/'report.html').write_text(page)
