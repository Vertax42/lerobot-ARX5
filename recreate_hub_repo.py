#!/usr/bin/env python3
"""
删除远程仓库并重新创建，然后 push 本地数据集
"""

from huggingface_hub import HfApi, create_repo
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 配置
REPO_ID = "Vertax/xense_bi_arx5_tie_shoelaces_tactile"
LOCAL_PATH = (
    "/home/vertax/.cache/huggingface/lerobot/Vertax/xense_bi_arx5_tie_shoelaces_tactile"
)

print(f"🗑️  准备删除并重建远程仓库...")
print(f"   Repo: {REPO_ID}")
print(f"   Local: {LOCAL_PATH}")
print()

api = HfApi()

# 1. 删除远程仓库
print("🗑️  删除远程仓库...")
try:
    api.delete_repo(repo_id=REPO_ID, repo_type="dataset")
    print("   ✓ 远程仓库已删除")
except Exception as e:
    print(f"   ⚠️  删除失败（可能不存在）: {e}")

print()

# 2. 重新创建仓库
print("📦 重新创建远程仓库...")
try:
    create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        exist_ok=True,
        private=False,  # 根据需要设置为 True 或 False
    )
    print("   ✓ 远程仓库已创建")
except Exception as e:
    print(f"   ℹ️  创建信息: {e}")

print()

# 3. 加载本地数据集
print("📂 加载本地数据集...")
dataset = LeRobotDataset(REPO_ID, root=LOCAL_PATH)

print(f"   ✓ 数据集信息:")
print(f"      - total_episodes: {dataset.meta.total_episodes}")
print(f"      - total_frames: {dataset.meta.total_frames}")
print(f"      - total_videos: {dataset.meta.info.get('total_videos', 'N/A')}")
print()

# 4. Push 到 Hub（使用 upload_large_folder 处理大数据集）
print("🚀 正在 push 到 Hub（大数据集上传，这可能需要较长时间）...")
print("   使用 upload_large_folder 方法分批上传...")
print("   ⚠️  首先清理远程仓库中的所有旧文件...")
try:
    # 使用 delete_patterns="*" 清理远程所有文件，然后重新上传
    dataset.push_to_hub(
        upload_large_folder=True, delete_patterns="*"  # 删除远程所有文件，强制重新上传
    )
    print("\n✅ 数据集已成功 push 到全新的 Hub 仓库！")
    print(f"\n🔗 查看: https://huggingface.co/datasets/{REPO_ID}")
except Exception as e:
    print(f"\n❌ Push 失败: {e}")
    raise
