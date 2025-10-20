#!/usr/bin/env python3
"""
删除指定的 episode 并更新所有 metadata。

用法:
    python delete_episode.py --dataset-path /path/to/dataset --episode-index 3
"""

import argparse
import json
import shutil
from pathlib import Path


def delete_episode(dataset_path: Path, episode_index: int):
    """删除指定的 episode 并更新 metadata"""

    print(f"🗑️  正在删除 Episode {episode_index}...")

    # 1. 删除 parquet 文件
    episode_chunk = episode_index // 1000  # 默认 chunks_size = 1000
    parquet_file = (
        dataset_path
        / f"data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    if parquet_file.exists():
        print(f"   ✓ 删除 parquet: {parquet_file}")
        parquet_file.unlink()
    else:
        print(f"   ⚠️  parquet 文件不存在: {parquet_file}")

    # 2. 删除视频文件
    videos_dir = dataset_path / f"videos/chunk-{episode_chunk:03d}"
    if videos_dir.exists():
        for camera_dir in videos_dir.iterdir():
            if camera_dir.is_dir():
                video_file = camera_dir / f"episode_{episode_index:06d}.mp4"
                if video_file.exists():
                    print(f"   ✓ 删除视频: {video_file}")
                    video_file.unlink()

    # 3. 删除图像文件夹（如果有）
    images_dir = dataset_path / f"images/chunk-{episode_chunk:03d}"
    if images_dir.exists():
        for camera_dir in images_dir.iterdir():
            if camera_dir.is_dir():
                image_folder = camera_dir / f"episode_{episode_index:06d}"
                if image_folder.exists():
                    print(f"   ✓ 删除图像文件夹: {image_folder}")
                    shutil.rmtree(image_folder)

    # 4. 读取 episodes.jsonl
    episodes_file = dataset_path / "meta/episodes.jsonl"
    episodes = []
    deleted_episode_length = 0

    with open(episodes_file, "r") as f:
        for line in f:
            ep = json.loads(line)
            if ep["episode_index"] != episode_index:
                episodes.append(ep)
            else:
                deleted_episode_length = ep["length"]
                print(
                    f"   📊 Episode {episode_index} 长度: {deleted_episode_length} 帧"
                )

    # 写回 episodes.jsonl（不包含被删除的 episode）
    with open(episodes_file, "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")
    print(f"   ✓ 更新 episodes.jsonl")

    # 5. 读取 episodes_stats.jsonl
    stats_file = dataset_path / "meta/episodes_stats.jsonl"
    stats_lines = []

    with open(stats_file, "r") as f:
        for line in f:
            stat = json.loads(line)
            if stat["episode_index"] != episode_index:
                stats_lines.append(line)

    # 写回 episodes_stats.jsonl
    with open(stats_file, "w") as f:
        for line in stats_lines:
            f.write(line)
    print(f"   ✓ 更新 episodes_stats.jsonl")

    # 6. 更新 info.json
    info_file = dataset_path / "meta/info.json"
    with open(info_file, "r") as f:
        info = json.load(f)

    old_total_episodes = info["total_episodes"]
    old_total_frames = info["total_frames"]
    old_total_videos = info["total_videos"]

    # 更新计数
    info["total_episodes"] -= 1
    info["total_frames"] -= deleted_episode_length
    info["total_videos"] -= 3  # 假设有 3 个相机

    # 更新 splits
    info["splits"]["train"] = f"0:{info['total_episodes']}"

    # 写回 info.json
    with open(info_file, "w") as f:
        json.dump(info, f, indent=4)

    print(f"   ✓ 更新 info.json:")
    print(f"      - total_episodes: {old_total_episodes} → {info['total_episodes']}")
    print(f"      - total_frames: {old_total_frames} → {info['total_frames']}")
    print(f"      - total_videos: {old_total_videos} → {info['total_videos']}")
    print(f"      - splits: {info['splits']}")

    print(f"\n✅ Episode {episode_index} 已成功删除！")
    print(
        f"\n⚠️  注意: 被删除的 episode 索引号 {episode_index} 不会被后续 episode 重用。"
    )
    print(f"   下次录制新 episode 时，索引会从 {old_total_episodes} 开始。")


def main():
    parser = argparse.ArgumentParser(description="删除 LeRobot 数据集中的指定 episode")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/home/vertax/.cache/huggingface/lerobot/Vertax/xense_bi_arx5_tie_shoelaces",
        required=True,
        help="数据集路径（例如: /home/vertax/.cache/huggingface/lerobot/Vertax/xense_bi_arx5_tie_shoelaces）",
    )
    parser.add_argument(
        "--episode-index", type=int, required=True, help="要删除的 episode 索引"
    )

    args = parser.parse_args()
    dataset_path = Path(args.dataset_path)

    if not dataset_path.exists():
        print(f"❌ 错误: 数据集路径不存在: {dataset_path}")
        return

    # 确认操作
    print(f"\n⚠️  警告: 即将删除以下数据集的 Episode {args.episode_index}:")
    print(f"   数据集路径: {dataset_path}")
    print(f"   Episode 索引: {args.episode_index}")

    response = input("\n确认删除吗？ (yes/no): ")
    if response.lower() != "yes":
        print("❌ 操作已取消")
        return

    delete_episode(dataset_path, args.episode_index)


if __name__ == "__main__":
    main()
