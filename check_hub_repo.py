#!/usr/bin/env python3
"""
检查远程 Hub 仓库的文件列表
"""

from huggingface_hub import HfApi

# 配置
REPO_ID = "Vertax/xense_bi_arx5_tie_shoelaces"

print(f"🔍 检查远程 Hub 仓库: {REPO_ID}")
print()

api = HfApi()

# 获取仓库文件列表
print("📂 获取文件列表...")
files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")

# 过滤 episode 相关文件
episode_files = [f for f in files if "episode_" in f and ("parquet" in f or "mp4" in f)]
episode_files.sort()

print(f"\n✅ 找到 {len(episode_files)} 个 episode 相关文件：")
print()

# 按类型分组
parquet_files = [f for f in episode_files if "parquet" in f]
video_files = [f for f in episode_files if "mp4" in f]

print("📊 Parquet 数据文件:")
for f in parquet_files:
    print(f"   - {f}")

print()
print("🎥 视频文件:")
video_by_episode = {}
for f in video_files:
    # 提取 episode 索引
    if "episode_" in f:
        ep_idx = f.split("episode_")[1].split(".")[0]
        if ep_idx not in video_by_episode:
            video_by_episode[ep_idx] = []
        video_by_episode[ep_idx].append(f)

for ep_idx in sorted(video_by_episode.keys()):
    print(f"   Episode {int(ep_idx)}:")
    for f in sorted(video_by_episode[ep_idx]):
        camera = f.split("/")[-2] if "/" in f else "unknown"
        print(f"      - {camera}")

# 检查 episode 3
print()
episode_3_files = [f for f in episode_files if "episode_000003" in f]
if episode_3_files:
    print("❌ 警告: Episode 3 的文件仍然存在于 Hub:")
    for f in episode_3_files:
        print(f"   - {f}")
else:
    print("✅ 确认: Episode 3 已从 Hub 删除!")

# 检查 metadata
print()
print("📋 Metadata 文件:")
meta_files = [f for f in files if f.startswith("meta/")]
for f in sorted(meta_files):
    print(f"   - {f}")

# 下载并检查 info.json
print()
print("📄 检查 info.json 内容:")
import json
from huggingface_hub import hf_hub_download

info_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="meta/info.json",
    repo_type="dataset",
    force_download=True,  # 强制重新下载，不使用缓存
)

with open(info_path, "r") as f:
    info = json.load(f)

print(f"   - total_episodes: {info['total_episodes']}")
print(f"   - total_frames: {info['total_frames']}")
print(f"   - total_videos: {info['total_videos']}")
print(f"   - splits: {info['splits']}")

print()
print("🔗 查看仓库: https://huggingface.co/datasets/" + REPO_ID)
