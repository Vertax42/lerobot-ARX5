#!/usr/bin/env python3
"""
将本地数据集 push 到 Hugging Face Hub
这会自动同步所有更改（包括删除的 episode）
"""

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 配置
REPO_ID = "Vertax/xense_bi_arx5_tie_white_shoelaces_1030_no_adjust"
LOCAL_PATH = "/home/vertax/.cache/huggingface/lerobot/Vertax/xense_bi_arx5_tie_white_shoelaces_1030_no_adjust"

print(f"📤 准备 push 数据集到 Hub...")
print(f"   Repo: {REPO_ID}")
print(f"   Local: {LOCAL_PATH}")
print()

# 加载本地数据集
print("📂 加载本地数据集...")
dataset = LeRobotDataset(REPO_ID, root=LOCAL_PATH)

print(f"   ✓ 数据集信息:")
print(f"      - total_episodes: {dataset.meta.total_episodes}")
print(f"      - total_frames: {dataset.meta.total_frames}")
print(f"      - total_videos: {dataset.meta.info.get('total_videos', 'N/A')}")
print()

# Push 到 Hub
print("🚀 正在 push 到 Hub（这可能需要几分钟）...")
try:
    dataset.push_to_hub()
    print("\n✅ 数据集已成功 push 到 Hub！")
    print(f"\n🔗 查看: https://huggingface.co/datasets/{REPO_ID}")
except Exception as e:
    print(f"\n❌ Push 失败: {e}")
    raise
