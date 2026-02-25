# Changelog

## [Unreleased]

### Added — Streaming Video Encoding + Hardware Encoder Support

Ported and extended from lerobot upstream PR #2974.

#### Core feature (`src/lerobot/datasets/video_utils.py`)

- **`HW_ENCODERS`** — ordered list of supported hardware encoder names:
  `h264_videotoolbox`, `hevc_videotoolbox`, `h264_nvenc`, `hevc_nvenc`,
  `h264_vaapi`, `h264_qsv`
- **`VALID_VIDEO_CODECS`** — set of all accepted codec names including `"auto"`
- **`detect_available_hw_encoders()`** — probes the system and returns a list
  of actually available hardware encoders
- **`resolve_vcodec(vcodec)`** — validates codec name; `"auto"` resolves to the
  first available hardware encoder, falling back to `libsvtav1`
- **`_get_codec_options(vcodec, g, crf, preset)`** — builds codec-specific PyAV
  option dicts (nvenc → `rc=constqp/qp`, vaapi → `qp`, videotoolbox → `q:v`,
  software → `g/crf`)
- **`_CameraEncoderThread`** — background `threading.Thread` per camera; reads
  frames from a queue, encodes to MP4 with PyAV, computes per-channel pixel
  statistics inline using `RunningQuantileStats`
- **`StreamingVideoEncoder`** — orchestrates per-camera encoder threads with a
  clean lifecycle API: `start_episode()`, `feed_frame()`, `finish_episode()`,
  `cancel_episode()`, `close()`
- Updated **`encode_video_frames()`** to accept `encoder_threads` and use
  `resolve_vcodec` / `_get_codec_options` instead of hardcoded logic
- Updated **`VideoEncodingManager.__exit__()`** to delegate to streaming
  encoder when active, preserving original batch-encode path otherwise

#### Dataset integration (`src/lerobot/datasets/lerobot_dataset.py`)

- `LeRobotDataset.create()` and `__init__()` accept four new parameters:
  - `vcodec` (default: `"libsvtav1"`)
  - `streaming_encoding` (default: `False`)
  - `encoder_queue_maxsize` (default: `30`)
  - `encoder_threads` (default: `None` — auto)
- `add_frame()`: starts encoding threads on the first frame of each episode;
  feeds video frames to the streaming encoder in place of PNG writing
- `save_episode()`: collects encoded MP4 paths and per-channel stats from
  encoder threads; computes `compute_episode_stats` only on non-video features
- `_save_episode_video()`: accepts optional `temp_path` to skip re-encoding
  when streaming encoder has already produced the file
- `clear_episode_buffer()`: cancels in-progress streaming episode on re-record
- Resume mode (`__init__`) fully supports streaming encoding — no fallback

#### CLI (`src/lerobot/scripts/lerobot_record.py`)

New `DatasetRecordConfig` fields:

| Field | Default | Description |
|---|---|---|
| `vcodec` | `"libsvtav1"` | Codec name or `"auto"` |
| `streaming_encoding` | `False` | Enable real-time background encoding |
| `encoder_queue_maxsize` | `30` | Max frames buffered per camera |
| `encoder_threads` | `None` | CPU threads per encoder (None = auto) |

#### Documentation

- New guide: `docs/source/streaming_video_encoding.mdx` — covers Quick Start,
  codec selection, performance tuning, recommended configurations per system
  type, architecture overview, and troubleshooting
- `docs/source/_toctree.yml` — registered under Datasets section

#### Tests

- `test/test_streaming_video_encoder.py` — 46 tests across 5 classes:
  - `TestGetCodecOptions` (9 tests)
  - `TestHWEncoderDetection` (7 tests)
  - `TestCameraEncoderThread` (5 tests)
  - `TestStreamingVideoEncoder` (18 tests)
  - `TestStreamingEncoderIntegration` (7 tests)

### Changed

- All `lerobot-record` commands in `client_commands.md` updated to include
  `--dataset.streaming_encoding=true --dataset.vcodec=auto`

### Notes

- Streaming encoding is **opt-in** (`streaming_encoding=False` by default);
  omitting the flag preserves identical behavior to the previous PNG→MP4 path
- `vcodec=auto` on this machine resolves to `h264_nvenc` (NVIDIA GPU detected)
- Upstream PR #2974 does not support resume mode; this implementation adds it
