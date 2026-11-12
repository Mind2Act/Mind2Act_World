"""RGB video output through an explicit local ffmpeg executable (no downloads)."""
from pathlib import Path
import subprocess
from mind2act_eval.timing import phases_for


DEFAULT_VIDEO_HZ = 6


def video_interval(physics_hz, video_hz):
    if type(video_hz) is not int or video_hz <= 0 or physics_hz % video_hz:
        raise ValueError("video_hz must be a positive integer divisor of physics_hz")
    return physics_hz // video_hz


class RGBRecorder:
    def __init__(self, output: Path, fps: int, *, ffmpeg="ffmpeg"):
        self.output = Path(output)
        self.fps = fps
        self.ffmpeg = ffmpeg
        self._writers = {}
        self.artifacts = {}
        self.frame_counts = {}

    def prepare(self, camera_sizes):
        """Start configured encoders before the native worker's large heap.

        First-frame writes use the same encoders/arguments. In particular,
        they no longer fork a second image of an already large Isaac worker.
        """
        for name, size in camera_sizes.items():
            self._writer(name, tuple(size))

    def _writer(self, name, size):
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise ValueError("camera name must be a safe artifact identifier")
        if len(size) != 2 or any(type(n) is not int or n <= 0 for n in size):
            raise ValueError("video size must contain positive integer width and height")
        if name not in self._writers:
            path = self.output / (name + ".mp4")
            log = (self.output / (name + "-ffmpeg.log")).open("wb")
            try:
                child = subprocess.Popen([self.ffmpeg, "-nostdin", "-n", "-f", "rawvideo",
                    "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(self.fps),
                    "-i", "pipe:0", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
            except BaseException:
                log.close()
                raise
            self._writers[name] = (child, log, size)
            self.artifacts["video_" + name] = path.name
        child, _, expected = self._writers[name]
        if size != expected:
            raise ValueError("camera resolution changed during episode")
        return child

    def capture(self, rgb):
        import numpy as np
        for name, frame in rgb.items():
            if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
                raise ValueError("camera name must be a safe artifact identifier")
            pixels = np.asarray(frame)
            if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3 or not pixels.shape[0] or not pixels.shape[1]:
                raise ValueError("video requires nonempty uint8 RGB HWC frames")
            size = (pixels.shape[1], pixels.shape[0])
            child = self._writer(name, size)
            with phases_for(self).measure('rgb.pack.' + name):
                payload = np.ascontiguousarray(pixels).tobytes()
            with phases_for(self).measure('pipe.write.' + name):
                child.stdin.write(payload)
            self.frame_counts[name] = self.frame_counts.get(name, 0) + 1

    def close(self):
        errors = []
        for child, log, _ in self._writers.values():
            try:
                child.stdin.close()
                try:
                    code = child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                    raise RuntimeError("video encoder did not close")
                if code:
                    raise RuntimeError("video encoder failed; see per-camera ffmpeg log")
            except Exception as exc:
                errors.append(str(exc))
            finally:
                log.close()
        self._writers.clear()
        if errors:
            raise RuntimeError("; ".join(errors))
