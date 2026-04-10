#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check the integrity and completeness of a local LeRobotDataset (v3.0).

Verifies:
  - meta/info.json exists and is valid
  - meta/stats.json exists
  - meta/tasks.parquet exists
  - episodes metadata matches info.json episode count
  - data parquet files: row count, index continuity, frame_index continuity, NaN check
  - video files: existence, frame count alignment with parquet, fps, resolution

Examples:

```shell
python src/lerobot/scripts/lerobot_check_dataset.py --repo-id Xense/assemble_box_with_phone_stand  --root ~/.cache/huggingface/lerobot
python lerobot-check-dataset.py --repo-id Xense/assemble_box_with_phone_stand
lerobot-check-dataset --repo-id Xense/assemble_box_with_phone_stand --root ~/.cache/huggingface/lerobot
lerobot-check-dataset --repo-id Xense/assemble_box_with_phone_stand --episode-index 0 2 4
```
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.utils.constants import HF_LEROBOT_HOME

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PASS = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check(
    cond: bool,
    msg_ok: str,
    msg_fail: str,
    errors: list,
    warnings: list,
    warn: bool = False,
) -> bool:
    if cond:
        logger.info("  %s %s", PASS, msg_ok)
        return True
    else:
        tag = WARN if warn else FAIL
        logger.warning("  %s %s", tag, msg_fail)
        if warn:
            warnings.append(msg_fail)
        else:
            errors.append(msg_fail)
        return False


def _ffprobe_stream(path: Path) -> dict:
    """Return the first video stream info dict from ffprobe, or empty dict on failure."""
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return {}
    try:
        data = json.loads(r.stdout)
        streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
        return streams[0] if streams else {}
    except (json.JSONDecodeError, IndexError):
        return {}


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------


def check_meta(dataset_root: Path, errors: list, warnings: list) -> dict:
    """Check meta/ directory and return parsed info.json (or {})."""
    logger.info("\n[meta]")
    meta_dir = dataset_root / "meta"

    info = {}
    info_path = meta_dir / "info.json"
    if not _check(
        info_path.exists(), "info.json found", "info.json missing", errors, warnings
    ):
        return info
    try:
        info = json.loads(info_path.read_text())
        logger.info("  %s info.json valid JSON", PASS)
    except json.JSONDecodeError as e:
        errors.append(f"info.json invalid JSON: {e}")
        logger.warning("  %s info.json invalid JSON: %s", FAIL, e)
        return info

    for fname in ("stats.json", "tasks.parquet"):
        _check(
            (meta_dir / fname).exists(),
            f"{fname} found",
            f"{fname} missing",
            errors,
            warnings,
        )

    return info


def check_episodes_meta(
    dataset_root: Path, info: dict, errors: list, warnings: list
) -> pd.DataFrame:
    """Check episodes parquet files under meta/episodes/ and return merged DataFrame."""
    logger.info("\n[meta/episodes]")
    ep_dir = dataset_root / "meta" / "episodes"

    ep_files = sorted(ep_dir.rglob("*.parquet")) if ep_dir.exists() else []
    _check(
        len(ep_files) > 0,
        f"{len(ep_files)} episode parquet file(s) found",
        "no episode parquet files found",
        errors,
        warnings,
    )

    frames: list[pd.DataFrame] = []
    for f in ep_files:
        try:
            frames.append(pq.read_table(f).to_pandas())
        except Exception as e:
            errors.append(f"Cannot read {f}: {e}")
            logger.warning("  %s Cannot read %s: %s", FAIL, f, e)

    if not frames:
        return pd.DataFrame()

    eps = pd.concat(frames, ignore_index=True)
    expected_total = info.get("total_episodes", -1)
    _check(
        len(eps) == expected_total,
        f"episode count matches info.json ({len(eps)})",
        f"episode count mismatch: found {len(eps)}, info.json says {expected_total}",
        errors,
        warnings,
    )
    return eps


def check_data(
    dataset_root: Path,
    info: dict,
    episode_indices: list[int] | None,
    errors: list,
    warnings: list,
) -> pd.DataFrame:
    """Check data parquet files and return merged DataFrame."""
    logger.info("\n[data]")
    data_dir = dataset_root / "data"

    data_files = sorted(data_dir.rglob("*.parquet")) if data_dir.exists() else []
    _check(
        len(data_files) > 0,
        f"{len(data_files)} data parquet file(s) found",
        "no data parquet files found",
        errors,
        warnings,
    )

    # Identify array features that must be stored as fixed_size_list in parquet.
    # Any feature with dtype=float32/float64 and a 1-D shape of length > 1 falls into this category.
    array_feature_cols = {
        k
        for k, v in info.get("features", {}).items()
        if v.get("dtype") in ("float32", "float64")
        and isinstance(v.get("shape"), list)
        and len(v["shape"]) == 1
        and v["shape"][0] > 1
    }

    schema_checked = (
        False  # report once for the first file; subsequent files would be identical
    )
    frames: list[pd.DataFrame] = []
    for f in data_files:
        try:
            table = pq.read_table(f)
            frames.append(table.to_pandas())
        except Exception as e:
            errors.append(f"Cannot read {f}: {e}")
            logger.warning("  %s Cannot read %s: %s", FAIL, f, e)
            continue

        if not schema_checked:
            schema_checked = True
            file_schema = table.schema
            meta = file_schema.metadata or {}
            has_pandas_meta = b"pandas" in meta
            _check(
                not has_pandas_meta,
                "parquet schema has no pandas metadata (HF viewer compatible)",
                "parquet schema contains pandas metadata — action/observation.state will appear "
                "as 'object' type in the HF viewer and line charts will not render "
                "(fix: re-merge with the current aggregate.py)",
                errors,
                warnings,
            )
            for col in sorted(array_feature_cols):
                if col not in file_schema.names:
                    continue
                col_type = file_schema.field(col).type
                is_fixed = pa.types.is_fixed_size_list(col_type)
                _check(
                    is_fixed,
                    f"{col}: parquet type is fixed_size_list (HF viewer compatible)",
                    f"{col}: parquet type is '{col_type}' instead of fixed_size_list — "
                    f"HF viewer and training will not display this column correctly "
                    f"(merge was done with an older aggregate.py)",
                    errors,
                    warnings,
                )

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)

    # Total frame count
    expected_frames = info.get("total_frames", -1)
    _check(
        len(data) == expected_frames,
        f"total frame count matches info.json ({len(data)})",
        f"total frame count mismatch: found {len(data)}, info.json says {expected_frames}",
        errors,
        warnings,
    )

    # Global index continuity
    idx_gaps = (data["index"].diff().dropna() != 1).sum()
    _check(
        idx_gaps == 0,
        "global index is continuous",
        f"global index has {idx_gaps} gap(s)",
        errors,
        warnings,
    )

    # Per-episode checks
    filter_eps = set(episode_indices) if episode_indices is not None else None
    for ep_idx, grp in data.groupby("episode_index"):
        if filter_eps is not None and ep_idx not in filter_eps:
            continue
        grp = grp.sort_values("frame_index")
        fi = list(grp["frame_index"].values)
        expected_fi = list(range(len(fi)))
        _check(
            fi == expected_fi,
            f"ep{ep_idx}: frame_index continuous ({len(fi)} frames)",
            f"ep{ep_idx}: frame_index not continuous",
            errors,
            warnings,
        )

    # NaN checks for action and observation.state
    for col in ("action", "observation.state"):
        if col not in data.columns:
            continue
        nan_count = (
            data[col]
            .apply(
                lambda x: any(v != v for v in x) if hasattr(x, "__iter__") else (x != x)
            )
            .sum()
        )
        _check(
            nan_count == 0,
            f"{col}: no NaN values",
            f"{col}: {nan_count} row(s) with NaN",
            errors,
            warnings,
        )

    return data


def check_data_file_refs(
    dataset_root: Path,
    eps_df: pd.DataFrame,
    errors: list,
    warnings: list,
) -> None:
    """Check that every data (chunk, file) referenced in episodes parquet exists on disk.

    This catches the merge bug where source datasets with many small data files are
    consolidated into fewer files but the episodes parquet still points to the old
    source file indices (e.g. file-003 through file-007 that no longer exist).
    """
    logger.info("\n[data file references]")

    if eps_df.empty:
        logger.info("  %s Skipping — no episodes metadata", WARN)
        return

    chunk_col = "data/chunk_index"
    file_col = "data/file_index"

    if chunk_col not in eps_df.columns or file_col not in eps_df.columns:
        logger.info(
            "  %s %s / %s columns missing from episodes parquet — skipping",
            WARN, chunk_col, file_col,
        )
        return

    unique_pairs = sorted({
        (int(c), int(f))
        for c, f in zip(eps_df[chunk_col], eps_df[file_col])
    })

    for chunk_idx, file_idx in unique_pairs:
        rel_path = f"data/chunk-{chunk_idx:03d}/file-{file_idx:03d}.parquet"
        _check(
            (dataset_root / rel_path).exists(),
            f"{rel_path} exists",
            f"{rel_path} referenced in episodes parquet but MISSING on disk "
            f"(merge may have remapped data files without updating episodes — "
            f"re-merge with the current aggregate.py)",
            errors,
            warnings,
        )


def check_videos(
    dataset_root: Path,
    info: dict,
    data: pd.DataFrame,
    episode_indices: list[int] | None,
    errors: list,
    warnings: list,
) -> None:
    """Check video files: existence, frame count, fps, resolution."""
    logger.info("\n[videos]")
    video_features = {
        k: v for k, v in info.get("features", {}).items() if v.get("dtype") == "video"
    }
    if not video_features:
        logger.info("  %s No video features defined in info.json", WARN)
        return

    video_path_template = info.get(
        "video_path",
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    )

    # Collect expected (video_key, chunk_index, file_index) from episodes metadata
    # We derive from the data parquet directly to stay self-contained
    # Group data by (chunk, file) per video key using episodes meta if available
    ep_meta_dir = dataset_root / "meta" / "episodes"
    ep_meta_files = (
        sorted(ep_meta_dir.rglob("*.parquet")) if ep_meta_dir.exists() else []
    )

    if ep_meta_files:
        ep_frames: list[pd.DataFrame] = []
        for f in ep_meta_files:
            try:
                ep_frames.append(pq.read_table(f).to_pandas())
            except Exception:
                pass
        eps = pd.concat(ep_frames, ignore_index=True) if ep_frames else pd.DataFrame()
    else:
        eps = pd.DataFrame()

    for video_key in video_features:
        chunk_col = f"videos/{video_key}/chunk_index"
        file_col = f"videos/{video_key}/file_index"

        if eps.empty or chunk_col not in eps.columns:
            # Fallback: scan videos directory
            video_dir = dataset_root / "videos" / video_key
            if not video_dir.exists():
                errors.append(f"Video directory missing: {video_dir}")
                logger.warning("  %s Video directory missing: %s", FAIL, video_dir)
                continue
            mp4_files = sorted(video_dir.rglob("*.mp4"))
            logger.info(
                "  %s %s: %d video file(s) found (frame count not verified — no episode metadata)",
                PASS if mp4_files else FAIL,
                video_key,
                len(mp4_files),
            )
            continue

        # Determine which video files to check (based on filtered episodes)
        if episode_indices is not None:
            ep_rows = eps.loc[eps["episode_index"].isin(episode_indices)]
        else:
            ep_rows = eps

        # Group by (chunk_index, file_index) — one check per video file
        file_groups = ep_rows.groupby([chunk_col, file_col])

        for (chunk_idx, file_idx), _ in file_groups:
            chunk_idx = int(chunk_idx)
            file_idx = int(file_idx)

            rel_path = video_path_template.format(
                video_key=video_key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )
            full_path = dataset_root / rel_path

            if not _check(
                full_path.exists(),
                f"{video_key}/chunk-{chunk_idx:03d}/file-{file_idx:03d}.mp4 exists",
                f"{video_key}/chunk-{chunk_idx:03d}/file-{file_idx:03d}.mp4 MISSING",
                errors,
                warnings,
            ):
                continue

            # ffprobe
            stream = _ffprobe_stream(full_path)
            if not stream:
                warnings.append(f"ffprobe failed for {rel_path}")
                logger.warning("  %s ffprobe failed: %s", WARN, rel_path)
                continue

            expected_fps = info.get("fps", 0)

            # fps check
            r_frame_rate = stream.get("r_frame_rate", "0/1")
            try:
                num, den = map(int, r_frame_rate.split("/"))
                actual_fps = num / den if den else 0
            except (ValueError, ZeroDivisionError):
                actual_fps = 0
            _check(
                abs(actual_fps - expected_fps) < 0.5,
                f"{video_key} file-{file_idx:03d}: fps={actual_fps:.0f}",
                f"{video_key} file-{file_idx:03d}: fps mismatch (expected {expected_fps}, got {actual_fps:.1f})",
                errors,
                warnings,
            )

            # Resolution check
            feat_info = video_features[video_key].get("info", {})
            exp_w = feat_info.get("video.width")
            exp_h = feat_info.get("video.height")
            act_w = stream.get("width")
            act_h = stream.get("height")
            if exp_w and exp_h:
                _check(
                    act_w == exp_w and act_h == exp_h,
                    f"{video_key} file-{file_idx:03d}: resolution {act_w}x{act_h}",
                    f"{video_key} file-{file_idx:03d}: resolution mismatch (expected {exp_w}x{exp_h}, got {act_w}x{act_h})",
                    errors,
                    warnings,
                )

            # Frame count: always sum ALL episodes in this file (not just filtered ones)
            # because the video file contains all of them regardless of which are being checked
            all_eps_in_file = eps.loc[
                (eps[chunk_col] == chunk_idx) & (eps[file_col] == file_idx)
            ]
            expected_frames_in_file = int(all_eps_in_file["length"].sum())
            nb_frames = int(stream.get("nb_frames", -1))
            ep_list = sorted(all_eps_in_file["episode_index"].tolist())
            _check(
                abs(nb_frames - expected_frames_in_file) <= len(all_eps_in_file) * 2,
                f"{video_key} file-{file_idx:03d}: {nb_frames} frames (expected ~{expected_frames_in_file}, eps={ep_list})",
                f"{video_key} file-{file_idx:03d}: frame count mismatch ({nb_frames} vs expected ~{expected_frames_in_file}, eps={ep_list})",
                errors,
                warnings,
                warn=True,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check integrity and completeness of a local LeRobotDataset.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Dataset repo id (e.g. Xense/assemble_box_with_phone_stand).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for local datasets. Defaults to HF_LEROBOT_HOME.",
    )
    parser.add_argument(
        "episodes_pos",
        type=int,
        nargs="*",
        metavar="EPISODE",
        help="Episode indices to check (positional, e.g. 0 1 2). Checks all if not specified.",
    )
    parser.add_argument(
        "--episode-index",
        "-e",
        type=int,
        nargs="+",
        default=None,
        metavar="IDX",
        dest="episode_index",
        help="Episode indices to check (named form, e.g. -e 0 1 2).",
    )
    args = parser.parse_args()

    # Merge positional and named episode index args; positional takes priority if both given
    if args.episodes_pos:
        episode_indices = args.episodes_pos
    elif args.episode_index:
        episode_indices = args.episode_index
    else:
        episode_indices = None

    root = args.root if args.root is not None else HF_LEROBOT_HOME
    dataset_root = root / args.repo_id

    logger.info("Checking dataset: %s", dataset_root)
    if not dataset_root.exists():
        logger.error("%s Dataset root not found: %s", FAIL, dataset_root)
        sys.exit(1)

    errors: list[str] = []
    warnings: list[str] = []

    info = check_meta(dataset_root, errors, warnings)
    eps_df = check_episodes_meta(dataset_root, info, errors, warnings)
    check_data_file_refs(dataset_root, eps_df, errors, warnings)
    data_df = check_data(dataset_root, info, episode_indices, errors, warnings)
    if not data_df.empty:
        check_videos(dataset_root, info, data_df, episode_indices, errors, warnings)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary: %d error(s), %d warning(s)", len(errors), len(warnings))
    if errors:
        logger.info("\nErrors:")
        for e in errors:
            logger.info("  %s %s", FAIL, e)
    if warnings:
        logger.info("\nWarnings:")
        for w in warnings:
            logger.info("  %s %s", WARN, w)

    if not errors:
        logger.info("\n%s Dataset integrity check passed.", PASS)
    else:
        logger.info("\n%s Dataset integrity check FAILED.", FAIL)
        sys.exit(1)


if __name__ == "__main__":
    main()
