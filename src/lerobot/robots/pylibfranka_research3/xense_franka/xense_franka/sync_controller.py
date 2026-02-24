"""
同步接口封装，内部使用异步实现。
允许用户不使用 async/await 语法控制机械臂。
"""
import asyncio
import threading
import numpy as np
from typing import Optional, List, Union
from scipy.spatial.transform import Rotation as R

from .robot import RobotInterface
from .controller import FrankaController


class SyncFrankaController:
    """
    同步的 Franka 控制器接口。
    
    示例用法:
    ```python
    from xense_franka import SyncFrankaController
    
    controller = SyncFrankaController("192.168.99.111")
    controller.start()
    controller.move([0, 0, 0, -1.57, 0, 1.57, 0.78])
    controller.switch("osc")
    
    for i in range(100):
        ee = controller.get_ee_pose()
        ee[:3, 3] += [0.001, 0, 0]
        controller.set_ee_pose(ee)
    
    controller.stop()
    ```
    """

    def __init__(self, robot_ip: str):
        """
        初始化同步控制器。
        
        Args:
            robot_ip: 机器人 IP 地址，如 "192.168.99.111"
        """
        self.robot_ip = robot_ip
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._robot: Optional[RobotInterface] = None
        self._controller: Optional[FrankaController] = None
        self._started = False

    def _run_loop(self):
        """在后台线程中运行事件循环"""
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        except SystemExit:
            pass  # 忽略控制器内部的 sys.exit

    def _run_async(self, coro, timeout=30.0):
        """在事件循环中运行协程并等待结果"""
        if not self._loop or not self._loop.is_running():
            # 如果循环没运行，尝试直接取消协程
            coro.close()
            return None
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except Exception:
            return None

    def start(self):
        """启动控制器和 1kHz 控制循环"""
        if self._started:
            return
        
        # 创建新的事件循环和后台线程
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        # 初始化机器人和控制器
        async def _init():
            self._robot = RobotInterface(self.robot_ip)
            self._controller = FrankaController(self._robot)
            await self._controller.start()

        self._run_async(_init())
        self._started = True

    def stop(self):
        """停止控制器"""
        if not self._started:
            return

        # 尝试优雅停止控制器
        if self._loop and self._loop.is_running():
            async def _stop():
                if self._controller:
                    try:
                        await self._controller.stop()
                    except Exception:
                        pass

            try:
                self._run_async(_stop(), timeout=3.0)
            except Exception:
                pass
            
            # 停止事件循环
            self._loop.call_soon_threadsafe(self._loop.stop)
        
        if self._thread:
            self._thread.join(timeout=2.0)
        self._started = False

    def move(self, q_target: Optional[List[float]] = None):
        """
        移动到目标关节位置（使用 Ruckig 轨迹规划）。
        
        Args:
            q_target: 目标关节角度（7个值），None 则移动到默认位置
        """
        self._run_async(self._controller.move(q_target))

    def switch(self, mode: str):
        """
        切换控制模式。
        
        Args:
            mode: "impedance" (关节阻抗) 或 "osc" (笛卡尔阻抗)
        """
        self._controller.switch(mode)

    def set_freq(self, freq: int):
        """
        设置 set 命令的更新频率。
        
        Args:
            freq: 频率 (Hz)，如 50
        """
        self._controller.set_freq(freq)

    def set_gains(self, kp: np.ndarray, kd: np.ndarray, mode: str = "osc"):
        """
        设置控制增益。
        
        Args:
            kp: 刚度增益
            kd: 阻尼增益
            mode: "osc" 或 "impedance"
        """
        if mode == "osc":
            self._controller.ee_kp = kp
            self._controller.ee_kd = kd
        else:
            self._controller.kp = kp
            self._controller.kd = kd

    def get_ee_pose(self) -> np.ndarray:
        """
        获取当前末端执行器位姿。
        
        Returns:
            4x4 齐次变换矩阵
        """
        with self._controller.state_lock:
            return self._controller.ee_desired.copy()

    def get_state(self) -> dict:
        """
        获取机器人完整状态。
        
        Returns:
            包含关节位置、速度、力矩等的字典
        """
        return self._robot.state

    def get_joint_positions(self) -> np.ndarray:
        """获取当前关节位置"""
        state = self._robot.state
        return np.array(state['q'])

    def get_joint_velocities(self) -> np.ndarray:
        """获取当前关节速度"""
        state = self._robot.state
        return np.array(state['dq'])

    def get_external_wrench(self) -> np.ndarray:
        """获取外部力/力矩"""
        state = self._robot.state
        return np.array(state['ext_wrench'])

    def set_ee_pose(self, pose: np.ndarray):
        """
        设置期望的末端执行器位姿（OSC 模式）。
        
        Args:
            pose: 4x4 齐次变换矩阵
        """
        self._run_async(self._controller.set("ee_desired", pose))

    def set_joint_positions(self, q: np.ndarray):
        """
        设置期望的关节位置（阻抗模式）。
        
        Args:
            q: 7 个关节角度
        """
        self._run_async(self._controller.set("q_desired", q))

    def move_delta(self, dx: float = 0, dy: float = 0, dz: float = 0,
                   drx: float = 0, dry: float = 0, drz: float = 0):
        """
        相对当前位置移动。
        
        Args:
            dx, dy, dz: 平移增量（米）
            drx, dry, drz: 旋转增量（度）
        """
        current_ee = self.get_ee_pose()
        
        # 应用平移
        current_ee[:3, 3] += np.array([dx, dy, dz])
        
        # 应用旋转
        if drx != 0 or dry != 0 or drz != 0:
            rotation_delta = R.from_euler('xyz', [drx, dry, drz], degrees=True).as_matrix()
            current_ee[:3, :3] = rotation_delta @ current_ee[:3, :3]
        
        self.set_ee_pose(current_ee)

    @property
    def initial_ee(self) -> np.ndarray:
        """获取初始末端位姿"""
        return self._controller.initial_ee.copy()

    @property
    def initial_qpos(self) -> np.ndarray:
        """获取初始关节位置"""
        return self._controller.initial_qpos.copy()

    def __enter__(self):
        """支持 with 语句"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持 with 语句"""
        self.stop()
        return False
