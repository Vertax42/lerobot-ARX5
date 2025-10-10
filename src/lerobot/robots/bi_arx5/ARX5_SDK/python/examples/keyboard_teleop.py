import time
from pynput import keyboard

from queue import Queue
import os
import sys

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
from arx5_interface import Arx5CartesianController, EEFState, Gain, LogLevel
from multiprocessing.managers import SharedMemoryManager

import time
import click


def start_keyboard_teleop(controller: Arx5CartesianController):

    ori_speed = 1.0
    pos_speed = 0.4
    gripper_speed = 0.2
    target_pose_6d = controller.get_home_pose()

    target_gripper_pos = 0.0
    cmd_dt = 0.01
    preview_time = 0.1
    window_size = 5
    keyboard_queue = Queue(window_size)
    robot_config = controller.get_robot_config()
    controller_config = controller.get_controller_config()

    print("Teleop tracking started.")

    key_pressed = {
        keyboard.Key.up: False,  # +x
        keyboard.Key.down: False,  # -x
        keyboard.Key.left: False,  # +y
        keyboard.Key.right: False,  # -y
        keyboard.Key.page_up: False,  # +z
        keyboard.Key.page_down: False,  # -z
        keyboard.KeyCode.from_char("q"): False,  # +roll
        keyboard.KeyCode.from_char("a"): False,  # -roll
        keyboard.KeyCode.from_char("w"): False,  # +pitch
        keyboard.KeyCode.from_char("s"): False,  # -pitch
        keyboard.KeyCode.from_char("e"): False,  # +yaw
        keyboard.KeyCode.from_char("d"): False,  # -yaw
        keyboard.KeyCode.from_char("r"): False,  # open gripper
        keyboard.KeyCode.from_char("f"): False,  # close gripper
        keyboard.Key.space: False,  # reset to home
    }

    def on_press(key):
        if key in key_pressed:
            key_pressed[key] = True

    def on_release(key):
        if key in key_pressed:
            key_pressed[key] = False

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    def get_filtered_keyboard_output(key_pressed: dict):
        state = np.zeros(6, dtype=np.float64)
        if key_pressed[keyboard.Key.up]:
            state[0] = 1
        if key_pressed[keyboard.Key.down]:
            state[0] = -1
        if key_pressed[keyboard.Key.left]:
            state[1] = 1
        if key_pressed[keyboard.Key.right]:
            state[1] = -1
        if key_pressed[keyboard.Key.page_up]:
            state[2] = 1
        if key_pressed[keyboard.Key.page_down]:
            state[2] = -1
        if key_pressed[keyboard.KeyCode.from_char("q")]:
            state[3] = 1
        if key_pressed[keyboard.KeyCode.from_char("a")]:
            state[3] = -1
        if key_pressed[keyboard.KeyCode.from_char("w")]:
            state[4] = 1
        if key_pressed[keyboard.KeyCode.from_char("s")]:
            state[4] = -1
        if key_pressed[keyboard.KeyCode.from_char("e")]:
            state[5] = 1
        if key_pressed[keyboard.KeyCode.from_char("d")]:
            state[5] = -1

        if (
            keyboard_queue.maxsize > 0
            and keyboard_queue._qsize() == keyboard_queue.maxsize
        ):
            keyboard_queue._get()

        keyboard_queue.put(state)

        return np.mean(np.array(list(keyboard_queue.queue)), axis=0)

    directions = np.zeros(6, dtype=np.float64)
    start_time = time.monotonic()
    loop_cnt = 0
    while True:
        eef_state = controller.get_eef_state()
        print(
            f"Time elapsed: {time.monotonic() - start_time:.03f}s, x: {eef_state.pose_6d()[0]:.03f}, y: {eef_state.pose_6d()[1]:.03f}, z: {eef_state.pose_6d()[2]:.03f}",
            end="\r",
        )
        # keyboard state is in the format of (x y z roll pitch yaw)
        prev_directions = directions
        directions = np.zeros(7, dtype=np.float64)
        state = get_filtered_keyboard_output(key_pressed)
        key_open = key_pressed[keyboard.KeyCode.from_char("r")]
        key_close = key_pressed[keyboard.KeyCode.from_char("f")]
        key_space = key_pressed[keyboard.Key.space]

        if key_space:
            controller.reset_to_home()
            target_pose_6d = controller.get_home_pose()
            target_gripper_pos = 0.0
            loop_cnt = 0
            start_time = time.monotonic()
            continue
        elif key_open and not key_close:
            gripper_cmd = 1
        elif key_close and not key_open:
            gripper_cmd = -1
        else:
            gripper_cmd = 0

        target_pose_6d[:3] += state[:3] * pos_speed * cmd_dt
        target_pose_6d[3:] += state[3:] * ori_speed * cmd_dt
        target_gripper_pos += gripper_cmd * gripper_speed * cmd_dt
        if target_gripper_pos >= robot_config.gripper_width:
            target_gripper_pos = robot_config.gripper_width
        elif target_gripper_pos <= 0:
            target_gripper_pos = 0
        loop_cnt += 1
        while time.monotonic() < start_time + loop_cnt * cmd_dt:
            pass

        current_timestamp = controller.get_timestamp()
        eef_cmd = EEFState()
        eef_cmd.pose_6d()[:] = target_pose_6d
        eef_cmd.gripper_pos = target_gripper_pos
        eef_cmd.timestamp = current_timestamp + preview_time

        # 打印详细的eef_cmd数据
        print(f"\n=== EEF Command Details ===")
        print(f"Current timestamp: {current_timestamp:.6f}")
        print(f"Command timestamp: {eef_cmd.timestamp:.6f}")
        print(f"Preview time: {preview_time:.3f}")
        print(f"Pose 6D: {eef_cmd.pose_6d()}")
        print(
            f"  Position (x, y, z): [{eef_cmd.pose_6d()[0]:.6f}, {eef_cmd.pose_6d()[1]:.6f}, {eef_cmd.pose_6d()[2]:.6f}]"
        )
        print(
            f"  Orientation (roll, pitch, yaw): [{eef_cmd.pose_6d()[3]:.6f}, {eef_cmd.pose_6d()[4]:.6f}, {eef_cmd.pose_6d()[5]:.6f}]"
        )
        print(f"Gripper position: {eef_cmd.gripper_pos:.6f}")
        print(f"Gripper velocity: {eef_cmd.gripper_vel:.6f}")
        print(f"Gripper torque: {eef_cmd.gripper_torque:.6f}")
        print(f"Target pose 6D: {target_pose_6d}")
        print(f"Target gripper pos: {target_gripper_pos:.6f}")
        print(f"Keyboard state: {state}")
        print(f"Gripper cmd: {gripper_cmd}")
        print(f"========================\n")

        controller.set_eef_cmd(eef_cmd)


@click.command()
@click.argument("model")  # ARX arm model: X5 or L5
@click.argument("interface")  # can bus name (can0 etc.)
def main(model: str, interface: str):
    controller = Arx5CartesianController(model, interface)
    controller.reset_to_home()

    robot_config = controller.get_robot_config()
    gain = Gain(robot_config.joint_dof)
    controller.set_log_level(LogLevel.DEBUG)
    np.set_printoptions(precision=4, suppress=True)
    try:
        start_keyboard_teleop(controller)
    except KeyboardInterrupt:
        print(f"Teleop recording is terminated. Resetting to home.")
        controller.reset_to_home()
        controller.set_to_damping()


if __name__ == "__main__":
    main()
