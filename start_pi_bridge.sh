#!/usr/bin/env bash
# Copy this file to the robot Pi and run there after powering up the arm.
set -eo pipefail
if [[ "${1:-}" == --help ]]; then
  echo 'On the arm Pi: ./start_pi_bridge.sh [bind-IP, default 192.168.77.225] [pi_server.py path]'
  exit 0
fi
pi_bridge_host="${1:-192.168.77.225}"
pi_bridge_file="${2:-$HOME/mouse_control_server/pi_server.py}"
if [[ ! -f "$pi_bridge_file" ]]; then
  echo "Missing Pi bridge: $pi_bridge_file. Pass its installed path as argument 2." >&2
  exit 1
fi
if [[ -n "$(ss -H -ltn 'sport = :9000')" ]]; then
  echo 'Port 9000 already has a listener; leave the existing arm bridge running.'
  exit 0
fi
exec python3 -u "$pi_bridge_file" "$pi_bridge_host"
