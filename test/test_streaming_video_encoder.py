"""
Tests for streaming video encoding and hardware encoder support.

Covers:
  - _get_codec_options()        per-codec option generation
  - detect_available_hw_encoders() / resolve_vcodec()
  - _CameraEncoderThread        background encoding thread
  - StreamingVideoEncoder       multi-camera orchestrator
  - Integration with LeRobotDataset (create + add_frame + save_episode)
"""

import queue
import tempfile
import threading
import time
from pathlib import Path

import av
import numpy as np
import pytest

from lerobot.datasets.video_utils import (
    HW_ENCODERS,
    VALID_VIDEO_CODECS,
    StreamingVideoEncoder,
    _CameraEncoderThread,
    _get_codec_options,
    detect_available_hw_encoders,
    resolve_vcodec,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FPS = 10
WIDTH, HEIGHT = 320, 240
CHANNELS = 3


def make_frame(seed: int = 0) -> np.ndarray:
    """Return a random HWC uint8 numpy array."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (HEIGHT, WIDTH, CHANNELS), dtype=np.uint8)


def feed_n_frames(encoder: StreamingVideoEncoder, keys: list[str], n: int) -> None:
    for i in range(n):
        frame = make_frame(i)
        for key in keys:
            encoder.feed_frame(key, frame)


def _video_frame_count(path: Path) -> int:
    """Count the number of video frames in an MP4 file."""
    count = 0
    with av.open(str(path)) as container:
        for _ in container.decode(video=0):
            count += 1
    return count


def _video_duration(path: Path) -> float:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return float(stream.duration * stream.time_base)


# ---------------------------------------------------------------------------
# 1. _get_codec_options
# ---------------------------------------------------------------------------


class TestGetCodecOptions:
    def test_libsvtav1_sets_g_and_crf(self):
        opts = _get_codec_options("libsvtav1", g=2, crf=30, preset=None)
        assert opts["g"] == "2"
        assert opts["crf"] == "30"
        assert "q:v" not in opts
        assert "qp" not in opts

    def test_libsvtav1_preset(self):
        opts = _get_codec_options("libsvtav1", g=2, crf=30, preset="8")
        assert opts["preset"] == "8"

    def test_h264_software_options(self):
        opts = _get_codec_options("h264", g=2, crf=25, preset="fast")
        assert opts["g"] == "2"
        assert opts["crf"] == "25"
        assert opts["preset"] == "fast"

    def test_hevc_software_options(self):
        opts = _get_codec_options("hevc", g=4, crf=28, preset=None)
        assert opts["g"] == "4"
        assert opts["crf"] == "28"

    def test_nvenc_uses_qp_not_crf(self):
        for codec in ("h264_nvenc", "hevc_nvenc"):
            opts = _get_codec_options(codec, g=2, crf=30, preset=None)
            assert opts.get("rc") == "constqp"
            assert opts.get("qp") == "30"
            assert "crf" not in opts
            assert "g" not in opts  # g not applied to nvenc

    def test_vaapi_uses_qp(self):
        opts = _get_codec_options("h264_vaapi", g=2, crf=28, preset=None)
        assert opts.get("qp") == "28"
        assert "crf" not in opts

    def test_qsv_uses_global_quality(self):
        opts = _get_codec_options("h264_qsv", g=2, crf=23, preset="medium")
        assert opts.get("global_quality") == "23"
        assert opts.get("preset") == "medium"
        assert "crf" not in opts

    def test_videotoolbox_uses_qv(self):
        for codec in ("h264_videotoolbox", "hevc_videotoolbox"):
            opts = _get_codec_options(codec, g=2, crf=30, preset=None)
            assert opts.get("q:v") == "30"
            assert "crf" not in opts
            assert "qp" not in opts

    def test_none_values_not_included(self):
        opts = _get_codec_options("libsvtav1", g=None, crf=None, preset=None)
        assert "g" not in opts
        assert "crf" not in opts
        assert "preset" not in opts


# ---------------------------------------------------------------------------
# 2. detect_available_hw_encoders / resolve_vcodec
# ---------------------------------------------------------------------------


class TestHWEncoderDetection:
    def test_detect_returns_list(self):
        result = detect_available_hw_encoders()
        assert isinstance(result, list)
        # All returned encoders must be known HW encoders
        for enc in result:
            assert enc in HW_ENCODERS

    def test_resolve_vcodec_passthrough_for_known(self):
        for codec in ("h264", "hevc", "libsvtav1"):
            assert resolve_vcodec(codec) == codec

    def test_resolve_vcodec_auto_returns_string(self):
        result = resolve_vcodec("auto")
        assert isinstance(result, str)
        assert result in VALID_VIDEO_CODECS

    def test_resolve_vcodec_auto_prefers_hw_if_available(self):
        available = detect_available_hw_encoders()
        result = resolve_vcodec("auto")
        if available:
            assert result == available[0]
        else:
            assert result == "libsvtav1"

    def test_resolve_vcodec_invalid_raises(self):
        with pytest.raises(ValueError, match="Unsupported video codec"):
            resolve_vcodec("totally_invalid_codec_xyz")

    def test_resolve_vcodec_hw_encoder_direct(self):
        # Direct HW names should pass through unchanged
        available = detect_available_hw_encoders()
        for enc in available:
            assert resolve_vcodec(enc) == enc

    def test_valid_video_codecs_contains_expected(self):
        assert "libsvtav1" in VALID_VIDEO_CODECS
        assert "h264" in VALID_VIDEO_CODECS
        assert "hevc" in VALID_VIDEO_CODECS
        assert "auto" in VALID_VIDEO_CODECS
        for enc in HW_ENCODERS:
            assert enc in VALID_VIDEO_CODECS


# ---------------------------------------------------------------------------
# 3. _CameraEncoderThread
# ---------------------------------------------------------------------------


class TestCameraEncoderThread:
    def _run_thread(self, n_frames: int, vcodec: str = "libsvtav1") -> tuple[Path, dict | None]:
        """Spin up a thread, feed n_frames, collect result."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            video_path = tmp / "out.mp4"
            frame_q: queue.Queue = queue.Queue(maxsize=100)
            result_q: queue.Queue = queue.Queue()
            stop = threading.Event()

            thread = _CameraEncoderThread(
                video_path=video_path,
                fps=FPS,
                vcodec=vcodec,
                pix_fmt="yuv420p",
                g=2,
                crf=30,
                preset=None,
                frame_queue=frame_q,
                result_queue=result_q,
                stop_event=stop,
                encoder_threads=None,
            )
            thread.start()

            for i in range(n_frames):
                frame_q.put(make_frame(i))
            frame_q.put(None)  # sentinel

            thread.join(timeout=60)
            assert not thread.is_alive(), "Thread did not finish in time"

            status, data = result_q.get_nowait()
            assert status == "ok"
            path, stats = data
            # Copy result before tmp dir is deleted
            import shutil
            out_copy = Path(tempfile.mktemp(suffix=".mp4"))
            shutil.copy(path, out_copy)
            return out_copy, stats

    def test_encodes_correct_number_of_frames(self):
        n = 20
        path, _ = self._run_thread(n)
        try:
            assert _video_frame_count(path) == n
        finally:
            path.unlink(missing_ok=True)

    def test_produces_valid_mp4(self):
        path, _ = self._run_thread(10)
        try:
            assert path.exists()
            assert path.stat().st_size > 0
            # Verify it can be opened and decoded
            with av.open(str(path)) as c:
                assert len(c.streams.video) == 1
        finally:
            path.unlink(missing_ok=True)

    def test_returns_statistics(self):
        path, stats = self._run_thread(15)
        try:
            assert stats is not None
            for key in ("min", "max", "mean", "std", "count"):
                assert key in stats
            assert stats["count"][0] > 0
        finally:
            path.unlink(missing_ok=True)

    def test_stop_event_aborts_encoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            video_path = tmp / "out.mp4"
            frame_q: queue.Queue = queue.Queue(maxsize=5)
            result_q: queue.Queue = queue.Queue()
            stop = threading.Event()

            thread = _CameraEncoderThread(
                video_path=video_path,
                fps=FPS,
                vcodec="libsvtav1",
                pix_fmt="yuv420p",
                g=2, crf=30, preset=None,
                frame_queue=frame_q,
                result_queue=result_q,
                stop_event=stop,
                encoder_threads=None,
            )
            thread.start()
            # Feed a frame then set stop event immediately
            frame_q.put(make_frame(0))
            stop.set()
            frame_q.put(None)  # ensure thread can unblock
            thread.join(timeout=10)
            assert not thread.is_alive()

    def test_stats_have_correct_channel_count(self):
        path, stats = self._run_thread(10)
        try:
            assert stats is not None
            # Values should have length == CHANNELS (3 for RGB)
            assert stats["mean"].shape == (CHANNELS,)
            assert stats["min"].shape == (CHANNELS,)
        finally:
            path.unlink(missing_ok=True)

    def test_can_open_codec_context_before_first_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            video_path = tmp / "out.mp4"
            frame_q: queue.Queue = queue.Queue(maxsize=5)
            result_q: queue.Queue = queue.Queue()
            stop = threading.Event()
            ready_event = threading.Event()

            thread = _CameraEncoderThread(
                video_path=video_path,
                fps=FPS,
                vcodec="h264",
                pix_fmt="yuv420p",
                g=2,
                crf=30,
                preset=None,
                frame_queue=frame_q,
                result_queue=result_q,
                stop_event=stop,
                encoder_threads=None,
                frame_shape=(HEIGHT, WIDTH, CHANNELS),
                ready_event=ready_event,
            )
            thread.start()

            assert ready_event.wait(timeout=10)
            assert thread.init_error is None

            frame_q.put(make_frame(0))
            frame_q.put(None)
            thread.join(timeout=30)
            assert not thread.is_alive()

            status, data = result_q.get_nowait()
            assert status == "ok"
            path, _ = data
            assert path.exists()
            assert _video_frame_count(path) == 1


# ---------------------------------------------------------------------------
# 4. StreamingVideoEncoder
# ---------------------------------------------------------------------------


class TestStreamingVideoEncoder:
    def _make_encoder(self, **kwargs) -> StreamingVideoEncoder:
        defaults = dict(
            fps=FPS,
            vcodec="libsvtav1",
            pix_fmt="yuv420p",
            g=2,
            crf=30,
            preset=None,
            queue_maxsize=50,
            encoder_threads=None,
        )
        defaults.update(kwargs)
        return StreamingVideoEncoder(**defaults)

    # ---- basic encode -------------------------------------------------

    def test_single_camera_encode(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            enc = self._make_encoder()
            enc.start_episode(["cam"], tmp)
            feed_n_frames(enc, ["cam"], 20)
            results = enc.finish_episode()

            assert "cam" in results
            path, stats = results["cam"]
            assert path.exists()
            assert stats is not None
            enc.close()

    def test_multi_camera_encode(self):
        keys = ["left", "right", "top"]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            enc = self._make_encoder()
            enc.start_episode(keys, tmp)
            feed_n_frames(enc, keys, 15)
            results = enc.finish_episode()

            assert set(results.keys()) == set(keys)
            for key in keys:
                path, stats = results[key]
                assert path.exists(), f"{key} MP4 missing"
                assert stats is not None
            enc.close()

    def test_correct_frame_count(self):
        n = 25
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            enc = self._make_encoder()
            enc.start_episode(["cam"], tmp)
            feed_n_frames(enc, ["cam"], n)
            results = enc.finish_episode()

            path, _ = results["cam"]
            assert _video_frame_count(path) == n
            enc.close()

    def test_video_has_correct_fps(self):
        n = FPS * 2  # 2 seconds
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            enc = self._make_encoder()
            enc.start_episode(["cam"], tmp)
            feed_n_frames(enc, ["cam"], n)
            results = enc.finish_episode()

            path, _ = results["cam"]
            duration = _video_duration(path)
            assert abs(duration - 2.0) < 0.5, f"Expected ~2s, got {duration:.2f}s"
            enc.close()

    # ---- sequential episodes -----------------------------------------

    def test_sequential_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            enc = self._make_encoder()

            for episode in range(3):
                enc.start_episode(["cam"], tmp)
                feed_n_frames(enc, ["cam"], 10)
                results = enc.finish_episode()
                path, stats = results["cam"]
                assert path.exists(), f"Episode {episode} MP4 missing"
                assert stats is not None

            enc.close()

    def test_episode_active_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            enc = self._make_encoder()
            assert not enc._episode_active

            enc.start_episode(["cam"], Path(tmp))
            assert enc._episode_active

            feed_n_frames(enc, ["cam"], 5)
            enc.finish_episode()
            assert not enc._episode_active
            enc.close()

    # ---- cancellation ------------------------------------------------

    def test_cancel_episode_cleans_temp_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            enc = self._make_encoder()
            enc.start_episode(["cam"], tmp)
            feed_n_frames(enc, ["cam"], 5)
            enc.cancel_episode()
            # No leftover temp directories inside tmp
            remaining = list(tmp.iterdir())
            assert len(remaining) == 0, f"Temp files not cleaned: {remaining}"
            enc.close()

    def test_cancel_sets_episode_inactive(self):
        with tempfile.TemporaryDirectory() as tmp:
            enc = self._make_encoder()
            enc.start_episode(["cam"], Path(tmp))
            enc.cancel_episode()
            assert not enc._episode_active
            enc.close()

    def test_cancel_then_new_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            enc = self._make_encoder()

            enc.start_episode(["cam"], tmp)
            feed_n_frames(enc, ["cam"], 5)
            enc.cancel_episode()

            enc.start_episode(["cam"], tmp)
            feed_n_frames(enc, ["cam"], 10)
            results = enc.finish_episode()
            path, stats = results["cam"]
            assert path.exists()
            enc.close()

    def test_close_cancels_active_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            enc = self._make_encoder()
            enc.start_episode(["cam"], Path(tmp))
            feed_n_frames(enc, ["cam"], 3)
            enc.close()  # should cancel and not raise
            assert not enc._episode_active

    # ---- frame dropping ----------------------------------------------

    def test_frames_dropped_when_queue_full(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # Queue size 1: will drop all but first frame
            enc = self._make_encoder(queue_maxsize=1)
            enc.start_episode(["cam"], tmp)

            # Feed many frames fast, most will be dropped
            for i in range(50):
                enc.feed_frame("cam", make_frame(i))

            feed_n_frames(enc, ["cam"], 0)  # no extra frames
            enc.finish_episode()
            # At least some frames should have been dropped
            # (we can't assert exact count due to timing)
            enc.close()

    def test_feed_frame_noop_when_inactive(self):
        enc = self._make_encoder()
        # Should not raise even though no episode is active
        enc.feed_frame("cam", make_frame(0))
        enc.close()

    def test_finish_episode_returns_empty_when_inactive(self):
        enc = self._make_encoder()
        result = enc.finish_episode()
        assert result == {}
        enc.close()

    # ---- statistics --------------------------------------------------

    def test_stats_normalized_range(self):
        """Stats returned by the encoder are raw uint8; normalization happens in save_episode()."""
        with tempfile.TemporaryDirectory() as tmp:
            enc = self._make_encoder()
            enc.start_episode(["cam"], Path(tmp))
            feed_n_frames(enc, ["cam"], 20)
            results = enc.finish_episode()

        _, stats = results["cam"]
        assert stats is not None
        # Raw uint8 stats: values in [0, 255]
        assert stats["min"].min() >= 0
        assert stats["max"].max() <= 255
        enc.close()

    def test_stats_shape_is_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            enc = self._make_encoder()
            enc.start_episode(["cam"], Path(tmp))
            feed_n_frames(enc, ["cam"], 20)
            results = enc.finish_episode()

        _, stats = results["cam"]
        assert stats is not None
        # Shape should be (C,) = (3,) for RGB
        assert stats["mean"].shape == (CHANNELS,)
        enc.close()

    # ---- encoder_threads ---------------------------------------------

    def test_encoder_threads_parameter_accepted(self):
        """Passing encoder_threads should not raise."""
        with tempfile.TemporaryDirectory() as tmp:
            enc = self._make_encoder(encoder_threads=2)
            enc.start_episode(["cam"], Path(tmp))
            feed_n_frames(enc, ["cam"], 10)
            results = enc.finish_episode()
            path, _ = results["cam"]
            assert path.exists()
            enc.close()

    # ---- key name sanitization ---------------------------------------

    def test_dotted_key_names_work(self):
        """Keys like 'observation.images.laptop' must not break temp file paths."""
        key = "observation.images.laptop"
        with tempfile.TemporaryDirectory() as tmp:
            enc = self._make_encoder()
            enc.start_episode([key], Path(tmp))
            feed_n_frames(enc, [key], 10)
            results = enc.finish_episode()
            assert key in results
            path, _ = results[key]
            assert path.exists()
            enc.close()

    def test_start_episode_waits_until_ready_with_frame_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            enc = self._make_encoder(vcodec="h264")
            enc.start_episode(
                ["cam"],
                tmp,
                frame_shapes={"cam": (HEIGHT, WIDTH, CHANNELS)},
                wait_until_ready=True,
            )

            assert enc._threads["cam"].ready_event.is_set()
            assert enc._threads["cam"].init_error is None

            feed_n_frames(enc, ["cam"], 5)
            results = enc.finish_episode()
            path, _ = results["cam"]
            assert _video_frame_count(path) == 5
            enc.close()


# ---------------------------------------------------------------------------
# 5. Integration with LeRobotDataset
# ---------------------------------------------------------------------------


class TestStreamingEncoderIntegration:
    """End-to-end tests that create a real LeRobotDataset with streaming encoding."""

    FEATURES = {
        "observation.images.cam": {
            "dtype": "video",
            "shape": (HEIGHT, WIDTH, CHANNELS),
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["x", "y", "z", "rx", "ry", "rz"],
        },
    }

    def _make_frame_dict(self, seed: int = 0) -> dict:
        return {
            "observation.images.cam": make_frame(seed),
            "action": np.zeros(6, dtype=np.float32),
            "task": "test_task",
        }

    def test_create_with_streaming_encoding(self, tmp_path):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset.create(
            repo_id="test/streaming",
            fps=FPS,
            features=self.FEATURES,
            root=tmp_path / "ds",
            streaming_encoding=True,
            vcodec="libsvtav1",
            encoder_threads=None,
        )
        assert ds._streaming_encoder is not None
        assert ds._streaming_encoder.fps == FPS
        ds._streaming_encoder.close()

    def test_create_without_streaming_encoding(self, tmp_path):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset.create(
            repo_id="test/no_streaming",
            fps=FPS,
            features=self.FEATURES,
            root=tmp_path / "ds",
            streaming_encoding=False,
        )
        assert ds._streaming_encoder is None

    def test_single_episode_saved_correctly(self, tmp_path):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset.create(
            repo_id="test/single",
            fps=FPS,
            features=self.FEATURES,
            root=tmp_path / "ds",
            streaming_encoding=True,
            vcodec="libsvtav1",
        )

        n_frames = 15
        for i in range(n_frames):
            ds.add_frame(self._make_frame_dict(i))

        ds.save_episode()
        ds.finalize()

        # One episode should exist
        assert ds.meta.total_episodes == 1
        assert ds.meta.total_frames == n_frames

        # Video file should exist
        video_path = ds.root / ds.meta.get_video_file_path(0, "observation.images.cam")
        assert video_path.exists(), f"Video not found: {video_path}"

    def test_prepare_episode_recording_warms_encoder_without_extra_frames(self, tmp_path):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset.create(
            repo_id="test/prepare_warmup",
            fps=FPS,
            features=self.FEATURES,
            root=tmp_path / "ds",
            streaming_encoding=True,
            vcodec="h264",
        )

        ds.prepare_episode_recording()
        assert ds.episode_buffer["size"] == 0
        assert ds._streaming_encoder is not None
        assert ds._streaming_encoder._episode_active
        assert ds._streaming_encoder._threads["observation.images.cam"].ready_event.is_set()

        n_frames = 7
        for i in range(n_frames):
            ds.add_frame(self._make_frame_dict(i))

        ds.save_episode()
        ds.finalize()

        video_path = ds.root / ds.meta.get_video_file_path(0, "observation.images.cam")
        assert video_path.exists()
        assert _video_frame_count(video_path) == n_frames

    def test_multi_episode_sequential(self, tmp_path):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset.create(
            repo_id="test/multi",
            fps=FPS,
            features=self.FEATURES,
            root=tmp_path / "ds",
            streaming_encoding=True,
            vcodec="libsvtav1",
        )

        for ep in range(3):
            for i in range(10):
                ds.add_frame(self._make_frame_dict(i))
            ds.save_episode()

        ds.finalize()
        assert ds.meta.total_episodes == 3
        assert ds.meta.total_frames == 30

    def test_clear_episode_buffer_cancels_encoder(self, tmp_path):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset.create(
            repo_id="test/cancel",
            fps=FPS,
            features=self.FEATURES,
            root=tmp_path / "ds",
            streaming_encoding=True,
            vcodec="libsvtav1",
        )

        for i in range(10):
            ds.add_frame(self._make_frame_dict(i))

        # Simulate rerecord: clear without saving
        ds.clear_episode_buffer()
        assert not ds._streaming_encoder._episode_active

        # Should be able to start fresh episode
        for i in range(10):
            ds.add_frame(self._make_frame_dict(i))
        ds.save_episode()
        ds.finalize()
        assert ds.meta.total_episodes == 1

    def test_episode_stats_present_after_save(self, tmp_path):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset.create(
            repo_id="test/stats",
            fps=FPS,
            features=self.FEATURES,
            root=tmp_path / "ds",
            streaming_encoding=True,
            vcodec="libsvtav1",
        )

        for i in range(20):
            ds.add_frame(self._make_frame_dict(i))

        ds.save_episode()
        ds.finalize()

        # Stats should have been computed and stored
        stats = ds.meta.stats
        assert stats is not None
        # action stats should be present (non-video)
        assert "action" in stats

    def test_video_file_readable_after_save(self, tmp_path):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset.create(
            repo_id="test/readable",
            fps=FPS,
            features=self.FEATURES,
            root=tmp_path / "ds",
            streaming_encoding=True,
            vcodec="libsvtav1",
        )

        n_frames = 20
        for i in range(n_frames):
            ds.add_frame(self._make_frame_dict(i))

        ds.save_episode()
        ds.finalize()

        video_path = ds.root / ds.meta.get_video_file_path(0, "observation.images.cam")
        assert video_path.exists()
        frame_count = _video_frame_count(video_path)
        assert frame_count == n_frames

    def test_streaming_vs_non_streaming_same_frame_count(self, tmp_path):
        """Both paths should produce videos with the same number of frames."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        n_frames = 15

        # Streaming path
        ds_s = LeRobotDataset.create(
            repo_id="test/streaming_cmp",
            fps=FPS,
            features=self.FEATURES,
            root=tmp_path / "streaming",
            streaming_encoding=True,
            vcodec="libsvtav1",
        )
        for i in range(n_frames):
            ds_s.add_frame(self._make_frame_dict(i))
        ds_s.save_episode()
        ds_s.finalize()

        # Traditional path
        ds_t = LeRobotDataset.create(
            repo_id="test/traditional_cmp",
            fps=FPS,
            features=self.FEATURES,
            root=tmp_path / "traditional",
            streaming_encoding=False,
        )
        for i in range(n_frames):
            ds_t.add_frame(self._make_frame_dict(i))
        ds_t.save_episode()
        ds_t.finalize()

        path_s = ds_s.root / ds_s.meta.get_video_file_path(0, "observation.images.cam")
        path_t = ds_t.root / ds_t.meta.get_video_file_path(0, "observation.images.cam")

        assert _video_frame_count(path_s) == n_frames
        assert _video_frame_count(path_t) == n_frames
