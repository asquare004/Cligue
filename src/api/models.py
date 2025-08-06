"""
Pydantic models for API request and response validation.
Generic models for any type of video content.
"""
from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class ChatMessage(BaseModel):
    message: str


class ChatRequest(BaseModel):
    message: str


class VideoEvent(BaseModel):
    timestamp: str
    type: str
    description: str
    severity: str
    objects: List[str]


class VideoStatistics(BaseModel):
    total_events: int
    events_per_minute: float
    event_types: Dict[str, int]
    severity_distribution: Dict[str, int]
    duration_minutes: float


class VideoHighlights(BaseModel):
    highlights: List[str]


class UploadResponse(BaseModel):
    status: str
    video_duration: float
    events_detected: int
    summary: str
    events_by_type: Dict[str, List[VideoEvent]]
    key_highlights: List[str]
    statistics: VideoStatistics


class ChatResponse(BaseModel):
    response: str
    status: str


class StatusResponse(BaseModel):
    video_loaded: bool
    events_count: int
    has_events: bool
    vlm_available: bool


class VideoAnalysisResponse(BaseModel):
    overview: str
    timeline: List[Dict[str, Any]]
    events_by_type: Dict[str, List[VideoEvent]]
    key_highlights: List[str]
    statistics: VideoStatistics
