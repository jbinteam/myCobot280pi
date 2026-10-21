#!/usr/bin/env python3
"""Read-only collector for /object_follow/timing; never calls arm services."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=60.)
    parser.add_argument('--output', type=Path, default=Path('output/tracking_timing_live.json'))
    args = parser.parse_args()
    if not 0 < args.seconds < float('inf'):
        parser.error('--seconds must be finite and positive')
    import rclpy
    from std_msgs.msg import String
    rclpy.init()
    node = rclpy.create_node('tracking_timing_collector')
    snapshots = []
    def receive(message):
        snapshots.append(json.loads(message.data))
    subscription = node.create_subscription(String, '/object_follow/timing', receive, 10)
    began = time.monotonic()
    try:
        while rclpy.ok() and time.monotonic()-began < args.seconds:
            rclpy.spin_once(node, timeout_sec=max(0., min(.25, args.seconds-(time.monotonic()-began))))
    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.monotonic()-began
        node.destroy_node()
        rclpy.try_shutdown()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'elapsed_s': elapsed, 'snapshots': snapshots,
        'note': 'Each snapshot is a rolling window, not an independent sample. Do not average its percentiles.'}, indent=2)+'\n')
    if not snapshots:
        print('No timing messages received. Start the updated ROS controller first.')
        raise SystemExit(2)
    print(f'{len(snapshots)} snapshots saved to {args.output}')
    print('Latest rolling window (ms; nested stages overlap):')
    for name, row in snapshots[-1]['stages'].items():
        print(f"{name:42s} median {row['median_ms']:9.3f}  p95 {row['p95_ms']:9.3f}  n={row['window_count']}")


if __name__ == '__main__':
    main()
