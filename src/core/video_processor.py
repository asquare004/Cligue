"""
Video processing module for extracting frames and validating video files.
"""
import cv2
import numpy as np
from typing import Generator
from dataclasses import dataclass
from src.utils.config import MAX_VIDEO_DURATION, FPS_SAMPLE_RATE


@dataclass
class VideoFrame:
    frame: np.ndarray
    timestamp: float
    frame_number: int


class VideoProcessor:
    def __init__(self, fps_sample_rate: int = FPS_SAMPLE_RATE):
        self.fps_sample_rate = fps_sample_rate

    def extract_frames(self, video_path: str) -> Generator[VideoFrame, None, None]:
        """Extract frames at specified sample rate"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / self.fps_sample_rate)
        if frame_interval == 0:
            frame_interval = 1
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                yield VideoFrame(
                    frame=frame,
                    timestamp=timestamp,
                    frame_number=frame_count
                )

            frame_count += 1

        cap.release()

    def validate_video(self, video_path: str) -> dict:
        """Validate video meets requirements"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return {"valid": False, "error": "Cannot open video"}

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0

        cap.release()

        return {
            "valid": duration <= MAX_VIDEO_DURATION,
            "duration": duration,
            "fps": fps,
            "frame_count": frame_count,
            "error": "Video too long" if duration > MAX_VIDEO_DURATION else None
        }
