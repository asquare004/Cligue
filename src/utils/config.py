"""
Configuration settings for the Visual Understanding Chat Assistant.
"""

import os

# VLM Configuration
VLM_MODEL = os.getenv("VLM_MODEL", "llava:7b")  # Default to LLaVA 7B via Ollama
VLM_API_BASE = os.getenv("VLM_API_BASE", "http://localhost:11434") # Ollama API

# Video Processing Configuration
FPS_SAMPLE_RATE = 1  # Frames per second to sample from the video
MAX_VIDEO_DURATION = 120  # Maximum video duration in seconds

# Agent Configuration
AGENT_MEMORY_K = 10  # Number of past interactions to remember

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Frontend Configuration
STREAMLIT_THEME = "dark"
