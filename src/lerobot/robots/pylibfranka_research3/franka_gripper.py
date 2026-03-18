"""Non-blocking gripper control for pylibfranka.

All pylibfranka C++ calls (move, read_once, homing, …) hold the Python GIL,
so running them in a *thread* still blocks the main teleop loop.  The high-level
:class:`FrankaGripper` therefore spawns a **child process** for all gripper I/O
and communicates via shared memory — ``get_gripper_position()`` and
``set_gripper_position()`` never block the caller.

The lower-level :class:`AsyncGripper` is kept for standalone scripts and the
``if __name__ == "__main__"`` demo, but is **not** used at runtime by
:class:`FrankaGripper`.
"""

import concurrent.futures
import logging
import multiprocessing
import threading

from pylibfranka import Gripper, GripperState


# ─── Standalone worker function (runs in child process) ───────────

def _gripper_io_worker(
    ip: str,
    speed: float,
    init_open: bool,
    cached_width: multiprocessing.Value,
    max_width_out: multiprocessing.Value,
    pending_target: multiprocessing.Value,
    stop_event: multiprocessing.Event,
    ready_event: multiprocessing.Event,
    error_event: multiprocessing.Event,
) -> None:
    """Child-process entry point.  Owns the only ``pylibfranka.Gripper``
    connection so that its GIL-holding C++ calls cannot stall the main process.
    """
    try:
        gripper = Gripper(ip)
        gripper.homing()

        state = gripper.read_once()
        max_width_out.value = state.max_width

        if init_open:
            gripper.move(state.max_width, speed)

        cached_width.value = gripper.read_once().width
        ready_event.set()

        while not stop_event.is_set():
            # Grab latest target (latest-value-wins)
            target = pending_target.value
            if target >= 0:
                pending_target.value = -1.0  # consume
                try:
                    gripper.move(target, speed)
                except Exception:
                    pass

            # Refresh cached width
            try:
                cached_width.value = gripper.read_once().width
            except Exception:
                pass

            # If idle (no pending target), avoid busy-spinning
            if pending_target.value < 0:
                stop_event.wait(0.02)

    except Exception:
        error_event.set()
        ready_event.set()  # unblock main process


# ─── Low-level thread-based wrapper (kept for standalone scripts) ─

class GripperFuture:
    """Future for async gripper operations, matching franky's BoolFuture API."""

    def __init__(self, future: concurrent.futures.Future):
        self._future = future

    def wait(self, timeout=None) -> bool:
        try:
            self._future.result(timeout=timeout)
            return True
        except concurrent.futures.TimeoutError:
            return False

    def get(self) -> bool:
        return self._future.result()


class AsyncGripper:
    """Thread-based wrapper around pylibfranka.Gripper.

    **Note:** because pylibfranka holds the GIL during C++ calls, this wrapper
    still blocks the calling process.  Use :class:`FrankaGripper` (which spawns
    a child process) for latency-critical code such as the teleop loop.
    """

    def __init__(self, fci_hostname: str):
        self._gripper = Gripper(fci_hostname)
        self._lock = threading.Lock()
        self._current_future: GripperFuture | None = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    @property
    def gripper(self) -> Gripper:
        return self._gripper

    def _set_current_future(self, fn, *args, **kwargs) -> GripperFuture:
        with self._lock:
            if self._current_future is not None:
                self._current_future.wait()
            future = self._executor.submit(fn, *args, **kwargs)
            self._current_future = GripperFuture(future)
            return self._current_future

    # ── Synchronous methods ──────────────────────────────────────

    def move(self, width: float, speed: float) -> bool:
        return self._gripper.move(width, speed)

    def grasp(self, width, speed, force, epsilon_inner=0.005, epsilon_outer=0.005) -> bool:
        return self._gripper.grasp(width, speed, force, epsilon_inner, epsilon_outer)

    def homing(self) -> bool:
        return self._gripper.homing()

    def stop(self) -> bool:
        return self._gripper.stop()

    def open(self, speed: float) -> bool:
        return self._gripper.move(self._gripper.read_once().max_width, speed)

    def read_once(self) -> GripperState:
        return self._gripper.read_once()

    @property
    def width(self) -> float:
        return self._gripper.read_once().width

    @property
    def is_grasped(self) -> bool:
        return self._gripper.read_once().is_grasped

    @property
    def max_width(self) -> float:
        return self._gripper.read_once().max_width

    # ── Async methods ────────────────────────────────────────────

    def move_async(self, width: float, speed: float) -> GripperFuture:
        return self._set_current_future(self._gripper.move, width, speed)

    def grasp_async(self, width, speed, force, epsilon_inner=0.005, epsilon_outer=0.005) -> GripperFuture:
        return self._set_current_future(self._gripper.grasp, width, speed, force, epsilon_inner, epsilon_outer)

    def open_async(self, speed: float) -> GripperFuture:
        return self._set_current_future(self.open, speed)

    def homing_async(self) -> GripperFuture:
        return self._set_current_future(self._gripper.homing)

    def stop_async(self) -> GripperFuture:
        return self._set_current_future(self._gripper.stop)


# ─── High-level interface (child-process based) ──────────────────

class FrankaGripper:
    """High-level Franka gripper with normalized position control.

    All pylibfranka I/O runs in a **child process** so that its GIL-holding
    C++ calls never stall the main teleop loop.

    Position convention: 0.0 = open, 1.0 = closed.
    """

    config_class = None

    def __init__(self, config):
        from .config_franka_gripper import FrankaGripperConfig
        self.config_class = FrankaGripperConfig

        self._config = config
        self._gripper_ip = config.gripper_ip
        self._gripper_speed = config.gripper_speed
        self._gripper_force = config.gripper_force
        self._gripper_min_pos = config.gripper_min_pos
        self._gripper_max_pos = config.gripper_max_pos
        self._init_open = config.init_open

        self._is_connected = False
        self._logger = logging.getLogger(f"FrankaGripper-{self._gripper_ip}")

        # Shared memory for inter-process communication
        self._cached_width = multiprocessing.Value("d", 0.0)
        self._max_width_out = multiprocessing.Value("d", 0.0)
        self._pending_target = multiprocessing.Value("d", -1.0)  # <0 = no target
        self._stop_event = multiprocessing.Event()
        self._ready_event = multiprocessing.Event()
        self._error_event = multiprocessing.Event()
        self._process: multiprocessing.Process | None = None

    def connect(self) -> None:
        """Connect to the Franka gripper (spawns worker process)."""
        if self._is_connected:
            raise RuntimeError("FrankaGripper already connected")

        self._logger.info(f"Connecting to Franka gripper at {self._gripper_ip} ...")
        self._stop_event.clear()
        self._ready_event.clear()
        self._error_event.clear()
        self._pending_target.value = -1.0

        self._process = multiprocessing.Process(
            target=_gripper_io_worker,
            args=(
                self._gripper_ip,
                self._gripper_speed,
                self._init_open,
                self._cached_width,
                self._max_width_out,
                self._pending_target,
                self._stop_event,
                self._ready_event,
                self._error_event,
            ),
            daemon=True,
        )
        self._process.start()

        if not self._ready_event.wait(timeout=30.0):
            raise RuntimeError("Gripper worker did not become ready within 30 s")
        if self._error_event.is_set():
            raise RuntimeError("Gripper worker failed during initialisation")

        hw_max = self._max_width_out.value
        if hw_max > 0:
            self._gripper_max_pos = hw_max
            self._logger.info(f"Gripper max width from hardware: {hw_max:.4f} m")

        self._is_connected = True
        self._logger.info("Franka gripper connected.")

    def disconnect(self) -> None:
        """Disconnect from the Franka gripper (stops worker process)."""
        if not self._is_connected:
            return
        self._stop_event.set()
        if self._process is not None:
            self._process.join(timeout=3.0)
            if self._process.is_alive():
                self._process.kill()
            self._process = None
        self._is_connected = False
        self._logger.info("Franka gripper disconnected.")

    def get_gripper_position(self) -> float:
        """Get normalized gripper position [0.0=open, 1.0=closed].

        Reads from shared memory — never blocks.
        """
        if not self._is_connected:
            return 0.0
        width = max(self._gripper_min_pos, min(self._gripper_max_pos, self._cached_width.value))
        pos_range = self._gripper_max_pos - self._gripper_min_pos
        if pos_range <= 0:
            return 0.0
        closed_ratio = 1.0 - (width - self._gripper_min_pos) / pos_range
        return max(0.0, min(1.0, closed_ratio))

    def set_gripper_position(self, normalized_pos: float) -> None:
        """Set gripper position — never blocks (latest-value-wins).

        Args:
            normalized_pos: Target position [0.0=open, 1.0=closed]
        """
        if not self._is_connected:
            return
        normalized_pos = max(0.0, min(1.0, normalized_pos))
        target_width = self._gripper_min_pos + (1.0 - normalized_pos) * (
            self._gripper_max_pos - self._gripper_min_pos
        )
        target_width = max(self._gripper_min_pos, min(self._gripper_max_pos, target_width))
        self._pending_target.value = target_width


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="AsyncGripper demo")
    parser.add_argument("--ip", type=str, required=True, help="Robot IP address")
    parser.add_argument("--speed", type=float, default=0.5, help="Gripper speed [m/s]")
    args = parser.parse_args()

    gripper = AsyncGripper(args.ip)

    print("Homing...")
    gripper.homing()
    print(f"Max width: {gripper.max_width:.4f} m")

    print("Moving to 0.07 m (async)...")
    gripper.move_async(0.07, args.speed)

    print("Moving to 0.02 m (async)...")
    gripper.move_async(0.02, args.speed)

    print("Moving to 0.07 m (async)...")
    gripper.move_async(0.07, args.speed)

    print("Moving to 0.05 m (async)...")
    future = gripper.move_async(0.05, args.speed)

    while not future.wait(timeout=0.1):
        state = gripper.read_once()
        print(f"  width = {state.width:.4f} m")

    print(f"Move result: {future.get()}")

    time.sleep(0.5)

    print("Opening fully (async)...")
    future = gripper.open_async(args.speed)
    future.wait()
    print(f"Open result: {future.get()}")
