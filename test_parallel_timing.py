#!/usr/bin/env python3
"""
测试并行执行的时序
"""

import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加 lerobot 到 Python 路径
sys.path.insert(0, "/home/ubuntu/lerobot-ARX5/src")

from lerobot.robots.bi_arx5.config_bi_arx5 import BiARX5Config
from lerobot.robots.bi_arx5.bi_arx5 import BiARX5

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_parallel_timing():
    """测试并行操作的实际时序"""

    print("=" * 60)
    print("ARX5 双臂并行时序测试")
    print("=" * 60)
    print("此测试将:")
    print("1. 连接双臂机器人")
    print("2. 等待用户确认后进行并行 reset_to_home 测试")
    print("3. 测量并行效率")
    print("4. 分析是否真正并行执行")
    print()

    # 创建配置
    config = BiARX5Config(
        id="test_bi_arx5",
        left_arm_model="X5",
        left_arm_port="can1",
        right_arm_model="X5",
        right_arm_port="can3",
        log_level="INFO",
        use_multithreading=True,
    )
    config.cameras = {}  # 不使用摄像头以简化测试

    # 创建机器人
    robot = BiARX5(config)

    try:
        logger.info("=== 开始连接机器人 ===")
        start_time = time.perf_counter()
        robot.connect(go_to_home=False)  # 先不回零
        connect_time = time.perf_counter() - start_time
        logger.info(f"连接耗时: {connect_time:.2f}秒")
        logger.info("✓ 机器人连接成功，当前处于重力补偿模式")

        print()
        print("⚠️  注意: 接下来的测试会让机器人运动回到原点，请确保安全!")
        print()

        while True:
            user_input = (
                input("确认开始并行 reset_to_home 测试? (输入 'y' 继续, 'n' 退出): ")
                .strip()
                .lower()
            )
            if user_input == "y":
                print("\n开始并行 reset_to_home 测试...")
                break
            elif user_input == "n":
                print("测试已取消")
                return
            else:
                print("请输入 'y' 或 'n'")

        logger.info("=== 开始并行 reset_to_home 测试 ===")

        def test_left_reset():
            logger.info(f"[{time.strftime('%H:%M:%S')}] 左臂开始 reset_to_home")
            start = time.perf_counter()
            robot.left_arm.reset_to_home()
            duration = time.perf_counter() - start
            logger.info(
                f"[{time.strftime('%H:%M:%S')}] 左臂完成 reset_to_home，耗时: {duration:.2f}秒"
            )
            return "left", duration

        def test_right_reset():
            logger.info(f"[{time.strftime('%H:%M:%S')}] 右臂开始 reset_to_home")
            start = time.perf_counter()
            robot.right_arm.reset_to_home()
            duration = time.perf_counter() - start
            logger.info(
                f"[{time.strftime('%H:%M:%S')}] 右臂完成 reset_to_home，耗时: {duration:.2f}秒"
            )
            return "right", duration

        # 测试并行执行
        overall_start = time.perf_counter()

        fL = robot._exec_left.submit(test_left_reset)
        fR = robot._exec_right.submit(test_right_reset)

        results = []
        for future in as_completed([fL, fR]):
            side, duration = future.result()
            results.append((side, duration))

        overall_duration = time.perf_counter() - overall_start

        logger.info("=== 并行测试结果 ===")
        for side, duration in results:
            logger.info(f"{side}臂单独耗时: {duration:.2f}秒")
        logger.info(f"总体并行耗时: {overall_duration:.2f}秒")

        # 理论上，如果真正并行，总体时间应该接近单个臂的最长时间
        max_individual = max(duration for _, duration in results)
        efficiency = max_individual / overall_duration * 100
        logger.info(f"并行效率: {efficiency:.1f}% (100%表示完全并行)")

    except KeyboardInterrupt:
        logger.info("用户中断测试")
    except Exception as e:
        logger.error(f"测试失败: {e}")
    finally:
        try:
            robot.disconnect()
        except Exception as e:
            logger.error(f"测试失败: {e}")
            pass


if __name__ == "__main__":
    test_parallel_timing()
