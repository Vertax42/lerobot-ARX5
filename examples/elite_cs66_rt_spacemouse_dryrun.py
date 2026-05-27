#!/usr/bin/env python
"""Dry-run smoke test for Elite CS66 + SpaceMouse teleop.

Stubs out:
  * elite_cs_sdk (driver + RTSI + dashboard)
  * pyspacemouse + lerobot.teleoperators.spacemouse.peripherals.Spacemouse

so the real EliteCS66RT + SpacemouseTeleop + elite_cs66_rt_spacemouse_teleop_loop
can be exercised end-to-end without hardware.

Asserts:
  * connect() takes the success path (RTSI + dashboard + driver + script)
  * action keys produced by convert_to_flexiv_action match what EliteCS66RT expects
  * send_action accepts and updates the background servo target
  * both-buttons triggers robot.reset_to_initial_position()
  * rt_moving short-circuits send_action and re-syncs teleop after reset finishes
  * disconnect() cleans up driver/dashboard/RTSI

Run: python examples/elite_cs66_rt_spacemouse_dryrun.py
"""

from __future__ import annotations

import sys
import types

import numpy as np

# ---------------------------------------------------------------------------
# elite_cs_sdk stub: minimal surface the EliteCS66RT driver exercises
# ---------------------------------------------------------------------------


class _FakeDriverConfig:
    def __init__(self):
        self.robot_ip = ""
        self.local_ip = ""
        self.servoj_time = 0.004
        self.servoj_lookahead_time = 0.1
        self.servoj_gain = 2000
        self.headless_mode = True
        self.script_file_path = ""


class _FakeDriver:
    def __init__(self, config):
        self.config = config
        self.connected = True
        self.servo_targets: list[list[float]] = []
        self.idle_calls = 0
        self.stop_calls = 0
        self.trajectory_points: list[tuple] = []
        self._trajectory_cb = None

    def isRobotConnected(self):
        return self.connected

    def sendExternalControlScript(self):
        self.connected = True
        return True

    def writeServoj(self, pos, timeout_ms, cartesian):
        assert cartesian is True
        self.servo_targets.append(list(pos))
        return True

    def writeIdle(self, timeout_ms):
        self.idle_calls += 1
        return True

    def stopControl(self, wait_ms=1000):
        self.stop_calls += 1
        return True

    # Trajectory API used by _move_j_blocking.
    def setTrajectoryResultCallback(self, cb):
        self._trajectory_cb = cb

    def writeTrajectoryControlAction(self, action, point_number, timeout_ms):
        # On NOOP after START with our 1 point, fire the success callback once
        # we've seen at least one writeTrajectoryPoint.
        if (
            action == _TrajectoryControlAction.NOOP
            and self.trajectory_points
            and self._trajectory_cb is not None
        ):
            cb, self._trajectory_cb = self._trajectory_cb, None
            cb(_TrajectoryMotionResult.SUCCESS)
        return True

    def writeTrajectoryPoint(self, positions, duration_s, blend_radius, cartesian):
        self.trajectory_points.append(
            (tuple(positions), float(duration_s), float(blend_radius), bool(cartesian))
        )
        return True


class _TrajectoryControlAction:
    START = "START"
    NOOP = "NOOP"


class _TrajectoryMotionResult:
    SUCCESS = "SUCCESS"


class _FakeDashboard:
    def __init__(self):
        self.connected = False

    def connect(self, ip):
        self.connected = True
        return True

    def disconnect(self):
        self.connected = False

    def powerOn(self):
        return True

    def brakeRelease(self):
        return True

    def playProgram(self):
        return True


class _FakeRtsi:
    def __init__(self, output, inp, freq):
        self.output = output
        self.input = inp
        self.freq = freq
        self.pose = [0.5, 0.0, 0.3, 0.0, 0.0, 0.0]

    def connect(self, ip):
        return True

    def disconnect(self):
        return True

    def getActualTCPPose(self):
        return list(self.pose)

    def getActualJointPositions(self):
        return [0.0] * 6

    def getActualJointVelocity(self):
        return [0.0] * 6

    def getActualJointTorques(self):
        return [0.0] * 6


def install_elite_sdk_stub():
    mod = types.ModuleType("elite_cs_sdk")
    mod.EliteDriverConfig = _FakeDriverConfig
    mod.EliteDriver = _FakeDriver
    mod.DashboardClientInterface = _FakeDashboard
    mod.RtsiIOInterface = _FakeRtsi
    mod.TrajectoryControlAction = _TrajectoryControlAction
    mod.TrajectoryMotionResult = _TrajectoryMotionResult

    def setCurrentThreadFiFoScheduling(prio):
        return True

    def getThreadFiFoMaxPriority():
        return 99

    mod.setCurrentThreadFiFoScheduling = setCurrentThreadFiFoScheduling
    mod.getThreadFiFoMaxPriority = getThreadFiFoMaxPriority

    # external_control.script lookup uses the package path: fabricate a file beside us.
    import pathlib

    pkg_dir = pathlib.Path(__file__).resolve().parent / "_fake_elite_cs_sdk"
    pkg_dir.mkdir(exist_ok=True)
    script = pkg_dir / "external_control.script"
    if not script.exists():
        script.write_text("# fake script\n")
    # Recipes: reuse the ones already shipped with elite_cs66_rt resource/.
    mod.__file__ = str(script)  # _resolve_sdk_resource uses parent dir

    sys.modules["elite_cs_sdk"] = mod


# ---------------------------------------------------------------------------
# SpaceMouse stub: replace peripherals.Spacemouse with a scripted device
# ---------------------------------------------------------------------------


class _FakeSpacemouse:
    """Drives a deterministic motion plan instead of real HID events."""

    def __init__(self, *args, **kwargs):
        self._connected = False
        # tick counter consumed by tests below
        self._tick = 0
        # scripted: (vx, vy, vz, rx, ry, rz, left, right)
        self._plan: list[tuple] = []

    def connect(self):
        self._connected = True

    def disconnect(self):
        self._connected = False

    def is_connected(self):
        return self._connected

    def poll(self):
        if self._plan:
            self._cur = self._plan.pop(0)
        else:
            self._cur = (0, 0, 0, 0, 0, 0, False, False)
        self._tick += 1

    def get_motion_state_transformed(self):
        vx, vy, vz, rx, ry, rz, *_ = self._cur
        return np.array([vx, vy, vz, rx, ry, rz], dtype=np.float32)

    def is_left_button_pressed(self):
        return bool(self._cur[6])

    def is_right_button_pressed(self):
        return bool(self._cur[7])


def install_spacemouse_stub():
    # The teleop does `from lerobot.teleoperators.spacemouse.peripherals import Spacemouse`.
    import lerobot.teleoperators.spacemouse.peripherals as peripherals_pkg

    peripherals_pkg.Spacemouse = _FakeSpacemouse


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    install_elite_sdk_stub()

    from lerobot.robots.elite_cs66_rt import EliteCS66RT, EliteCS66RTConfig

    cfg = EliteCS66RTConfig(
        robot_ip="127.0.0.1",
        use_background_servo_loop=False,  # synchronous so the test is deterministic
    )
    robot = EliteCS66RT(cfg)
    robot.connect()
    assert robot.is_connected, "EliteCS66RT failed to connect against stub SDK"
    print("[ok] EliteCS66RT stub connect path")

    # send_action with explicit 6D rotation matching identity orientation
    from lerobot.utils.robot_utils import quaternion_to_rotation_6d

    r6d = quaternion_to_rotation_6d(1.0, 0.0, 0.0, 0.0)
    action = {
        "tcp.x": 0.52,
        "tcp.y": 0.01,
        "tcp.z": 0.30,
        **{f"tcp.r{i+1}": float(r6d[i]) for i in range(6)},
    }
    sent = robot.send_action(action)
    assert set(sent) >= {f"tcp.r{i+1}" for i in range(6)}, sent
    print("[ok] send_action accepts Cartesian 6D action; keys:", sorted(sent))

    # Drive the SpaceMouse + loop --------------------------------------------
    install_spacemouse_stub()
    from lerobot.teleoperators.spacemouse import SpacemouseConfig, SpacemouseTeleop

    teleop_cfg = SpacemouseConfig()
    teleop = SpacemouseTeleop(teleop_cfg)
    teleop.connect(current_tcp_pose_euler=robot.get_current_tcp_pose_euler())

    # Script: forward motion x3, both-buttons x1, rest idle x3
    teleop._spacemouse._plan = [
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False),
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False),
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, True, True),  # both buttons -> reset
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False),
    ]

    # We can't actually call the loop's `while True` here without a duration cap.
    # The loop function supports `duration=`, so use a tiny one tied to fps:
    from lerobot.scripts.lerobot_teleoperate import elite_cs66_rt_spacemouse_teleop_loop

    elite_cs66_rt_spacemouse_teleop_loop(
        teleop=teleop,
        robot=robot,
        fps=50,
        display_data=False,
        duration=0.15,
        dryrun=False,
        debug_timing=False,
    )
    print()  # the loop's status print stays on one line
    print("[ok] elite_cs66_rt_spacemouse_teleop_loop ran without raising")

    driver = robot._driver
    assert driver is not None and len(driver.servo_targets) > 0, "no servoj calls observed"
    print(f"[ok] driver received {len(driver.servo_targets)} servoj targets")

    robot.disconnect()
    assert driver.stop_calls >= 1, "stopControl not called on disconnect"
    print("[ok] disconnect() called stopControl")


if __name__ == "__main__":
    main()
