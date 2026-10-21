# Pi bridge for hardware mouse tracking

This arm runs Atom firmware 6.5. Its original `/home/er/Server_280.py` waits
about 0.5 seconds after every motor target and Stop even though those packets
return no acknowledgement. Its TCP loop also treats a whole `recv` result as
one command, so combined or fragmented packets are not handled separately.

`pi_server.py` keeps the original serial settings, logger and GPIO support. It
uses `pi_transport.py` to reconstruct complete packets, forward each command
once, match query replies, and skip waits for these asynchronous UI commands:
SET_FRESH_MODE, SEND_ANGLE, SEND_ANGLES, STOP, JOG_ANGLE and
SET_ENCODERS_DRAG (0x3E). Encoder/speed packets are sent once asynchronously;
waiting for their ACK would add 0.5 seconds before the next telemetry query. Mode changes remain
verified by GET_FRESH_MODE in the PC controller. A client disconnect sends Stop.
The original Pi server file is retained without modification.

Install both files in `/home/er/mouse_control_server/`, with the original
`/home/er/Server_280.py` present. Disconnect the UI and stop the previous TCP
server before running:

```bash
cd /home/er
python3 -u mouse_control_server/pi_server.py 192.168.77.225
```

Only one process may own `/dev/ttyAMA0` and TCP port 9000. The PC UI still runs
with `./start_webui.sh --hardware` and connects to the same address. This bridge
supports the UI's asynchronous motion calls; synchronous calls that expect
motor acknowledgements are not supported by this firmware.

The PC uses the SDK's original validation, packet encoder and reply decoder
with a framed reply reader and an 800 ms timeout. Reference integration remains
bounded to 150 ms per step; delayed cycles slow the reference instead of
creating larger motor jumps. A cycle exceeding two seconds, stale heartbeats,
failed health/mode queries or collision checks stops tracking.

Both framed readers require exactly 12 payload bytes for GET_ANGLES (six signed
16-bit angles). Short replies with the same command code are discarded while
waiting for complete telemetry. The PC error reports the SDK status, failed
reply attempts and malformed payload sizes, without substituting old angles.

If the UI loses joint telemetry, check the Pi address and the network first:
the configured address must be reachable on TCP port 9000, with the Pi bridge
running. A changed Wi-Fi network or Pi IP can surface as an SDK `-1` angle
result. Reconnect after restoring the link; restarting mouse planning alone
does not repair a missing telemetry connection.

The packet/SDK/Pi-loop tests use fake sockets and serial transport. They send
no physical position command. Physical tracking smoothness still requires an
operator trial through the UI.
# Joint command comparison

`../webui/server.py` exposes `POST /api/ik/joint-diagnostic/plan` for a bounded
single-joint comparison. Planning captures fresh live angles and checks the
straight path against the pinned self/table/joint-limit scene without sending
motion. Execution uses the existing plan-ID, confirmation, heartbeat, Stop,
mode-lease and live-segment checks. Only the chosen joint target changes, with
a maximum excursion of 3°. The diagnostic records a 2-second endpoint dwell;
reported angle differences remain subject to telemetry resolution and cadence.

From the project root, with an idle connected arm in hardware mode:

```bash
.venv/bin/python hardware/joint_command_comparison.py
```

This prepares the comparison and saves its geometry without moving the arm.
Adding `--execute` explicitly starts the physical comparison and return legs.
Defaults are J4, a +2° excursion and SDK speed 70. The second trial uses 0.02°
position increments with a minimum 0.3-second command interval. Actual polling
and validation can increase this interval. Data is saved to
`output/joint_command_comparison.json`; no session token is written.

The two trials share SDK speed and absolute endpoint, but have different
reference durations. After the return, the second measured starting pose can
differ by up to 0.5°; both captured starts and commanded excursions are saved.
They compare command granularity, not an independently
calibrated minimum motor velocity. Failures abort the sequence without retry;
WebUI Stop/Escape and the shared execution lease remain active.

# Direction-and-speed qualification

`POST /api/ik/jog-qualification/plan` prepares a read-only, fixed J4/J6 test.
It samples the ±4° two-joint rectangle against the pinned self/table/joint-limit
scene, including a margin around the monitored 2.5° excursion. This is a
sampled collision check, not a continuous physical stopping guarantee.

```bash
.venv/bin/python hardware/jog_qualification.py
.venv/bin/python hardware/jog_qualification.py --execute
```

The experimental executor was designed to send `jog_angle(J4, direction, 1)`, waits at least 0.8 seconds,
then sends `jog_angle(J6, direction, 1)` without stopping J4. After a further
0.8 seconds it sends global Stop and observes settling. Telemetry latency can
extend the phases; a 2.5-second controller watchdog also sends Stop. Directions
point toward each joint's range midpoint. Untested-joint drift, excessive
measured velocity, excursion, collision, refresh-mode loss, or heartbeat loss
stop the test. Existing movement/frame/scene confirmations and ownership apply.
There is no automatic retry or return motion.

The saved `output/jog_qualification.json` distinguishes simultaneous measured
movement from a second packet replacing the first, or reports insufficient
evidence. SDK setting 1 is recorded as an SDK value, never converted to a
calibrated velocity. A successful two-joint trial does not qualify six-axis
trajectory jogging, per-joint braking, or an SDK-to-degrees/second mapping.
Production fixed-target and mouse paths continue using their existing checked
position-command execution until those capabilities are established.


The 2026-10-06 physical trial failed its intended bounds: J4 moved from 39.90°
to 141.15°. A blocking health query delayed the shared-lock Stop watchdog;
the second jog packet was sent at 7.52 seconds before the watchdog acquired
the lock. Telemetry did not capture the large travel before Stop. The arm
subsequently reported idle. See `output/jog_qualification.json` for the raw
captured trace and incident summary. The trace does not establish calibrated
speed, direction behavior, or multi-joint support.

**Physical qualification execution is now blocked unconditionally**, including
`--execute`; read-only planning remains available. Manual directional jog is
also blocked on Atom 6.5 in the backend and WebUI. This guard has no user
runtime override. A local Pi-side watchdog able to stop without waiting for
remote telemetry, plus firmware packet validation, is required before any
further directional-jog trial. The worker also checks elapsed time after
blocking queries to prevent issuing a new jog after its deadline.

Streaming serial queries wait at most 150 ms per attempt (normal measured replies are about 9–17 ms). During fixed v2 velocity-profile execution, the PC waits at most 250 ms per SDK reply attempt; the original idle timeout is restored afterward. Missing telemetry still fails health checks, and the two-second cycle stop remains active. Failed execution status includes `failed_cycle_timing` with individual query outcomes and durations.

The arm also emits `FF FF` servo status/debug data on the serial line. A trailing `FE` checksum can overlap the next `FE FE` reply header. The packet reader slides across repeated sync bytes rather than treating a third `FE` as a 254-byte length (unsupported by this bridge). This prevents healthy replies from being held behind a false 257-byte frame. The regression checks include byte-split overlap cases and captured read-only serial replies.
