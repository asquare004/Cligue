"""
Utility functions for the Visual Understanding Chat Assistant.
"""

import base64
from datetime import timedelta
import cv2
import numpy as np


def format_timestamp(seconds: float) -> str:
    """Format timestamp as MM:SS"""
    td = timedelta(seconds=seconds)
    return str(td)[2:7]  # Remove hours, keep MM:SS


def frame_to_base64(frame: np.ndarray) -> str:
    """Convert a numpy array frame to a base64 string for Ollama."""
    # Convert BGR to RGB (OpenCV uses BGR by default)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Encode to JPEG
    _, buffer = cv2.imencode(".jpg", frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    # Convert to base64
    base64_string = base64.b64encode(buffer).decode("utf-8")
    
    return base64_string
