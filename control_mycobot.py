#!/usr/bin/env python3
"""Inspect or move a myCobot 280 Pi through its TCP server."""

import argparse
import math
import socket
import sys
import time

from pymycobot import MyCobot280Socket
from simulation.kinematics import ARM


# Bent pose from pymycobot/demo/myCobot_280_demo/280_draw_gcode.py.
# A custom rest preset; check clearance with the installed tool before use.
REST_ANGLES = [0, -40, -130, 80, 0, 50]


def wait_for_target(robot, target, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        angles = robot.get_angles()
        if isinstance(angles, list) and len(angles) == 6:
            if all(abs(actual - expected) <= 2 for actual, expected in zip(angles, target)):
                if robot.is_moving() == 0:
                    return angles
        time.sleep(0.5)
    raise RuntimeError("Target was not reached within 90 seconds; check the robot before retrying")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host", help="IP address of the robot's Raspberry Pi")
    parser.add_argument("--port", type=int, default=9000)
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("status", help="read firmware version and joint angles")
    joint = commands.add_parser("joint", help="move one joint after confirmation")
    joint.add_argument("number", type=int, choices=range(1, 7), metavar="1-6")
    joint.add_argument("angle", type=float, help="target angle in degrees")
    joint.add_argument("--speed", type=int, default=30, help="1 to 100 (default: 30)")
    zero = commands.add_parser("zero", help="move all six joints to 0 degrees after confirmation")
    zero.add_argument("--speed", type=int, default=30, help="1 to 100 (default: 30)")
    rest = commands.add_parser("rest", help="move to the custom folded rest pose after confirmation")
    rest.add_argument("--speed", type=int, default=30, help="1 to 100 (default: 30)")
    rest.add_argument("--preview", action="store_true", help="show target angles without connecting or moving")
    jog = commands.add_parser("jog", help="move one joint by direction and speed for a limited time")
    jog.add_argument("number", type=int, choices=range(1, 7), metavar="1-6")
    jog.add_argument("direction", choices=("positive", "negative"))
    jog.add_argument("--speed", type=int, default=30, help="1 to 100 (default: 30)")
    jog.add_argument("--duration", type=float, default=0.5, help="movement duration in seconds (default: 0.5)")
    args = parser.parse_args()
    if args.command == "joint":
        if not math.isfinite(args.angle):
            parser.error("the target angle must be a finite number")
        low=math.degrees(ARM.lower[args.number-1]);high=math.degrees(ARM.upper[args.number-1])
        if not low<=args.angle<=high:
            parser.error(f"Joint {args.number} operating range is {low:.1f} to {high:.1f} degrees")
    if args.command in ("joint", "zero", "rest", "jog"):
        if not 1 <= args.speed <= 100:
            parser.error("--speed must be between 1 and 100")
    if args.command == "jog" and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be a finite positive number")
    if args.command == "rest" and args.preview:
        print(f"Custom rest joint angles: {REST_ANGLES} degrees; speed {args.speed}")
        return 0

    socket.setdefaulttimeout(5)
    robot = None
    try:
        robot = MyCobot280Socket(args.host, args.port)
        if args.command in (None, "status"):
            print("Atom firmware:", robot.get_system_version())
            print("Joint angles:", robot.get_angles())
            return 0

        current = robot.get_angles()
        if not isinstance(current, list) or len(current) != 6:
            raise RuntimeError(f"Could not read the current joint angles: {current!r}")
        print("Current joint angles:", current)
        if args.command in ("zero", "rest"):
            target = [0] * 6 if args.command == "zero" else REST_ANGLES[:]
            print(f"All joints -> {target} degrees at speed {args.speed}")
            if args.command == "rest":
                print("Custom folded pose: check clearance around the arm, base, and tool.")
        elif args.command == "jog":
            print(f"Joint {args.number}: {args.direction} direction at speed {args.speed} for {args.duration} seconds")
        else:
            print(f"Joint {args.number}: {current[args.number - 1]}° -> {args.angle}° at speed {args.speed}")
        if input("Clear the robot's path, then type MOVE to continue: ") != "MOVE":
            print("Cancelled.")
            return 0
        if args.command == "jog":
            try:
                robot.jog_angle(args.number, int(args.direction == "positive"), args.speed, _async=True)
                time.sleep(args.duration)
            finally:
                robot.stop()
                print("Stop command sent.", flush=True)
            print("Joint angles:", robot.get_angles())
        elif args.command in ("zero", "rest"):
            robot.send_angles(target, args.speed, _async=True)
            print("Command sent; waiting for target angles...", flush=True)
            print("Target reached. Joint angles:", wait_for_target(robot, target))
        else:
            result = robot.send_angle(args.number, args.angle, args.speed)
            print("Command result:", result)
            print("Joint angles immediately after command (motion may still be in progress):", robot.get_angles())
        return 0
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError, EOFError) as exc:
        print(f"Robot connection/control failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if robot is not None:
            robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
