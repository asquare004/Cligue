"""
Event detection module for identifying events in video frames.
Generic implementation for any type of video content.
"""
from typing import Dict, List, Any
import re
from dataclasses import dataclass
from enum import Enum
from src.core.vlm_interface import VLMInterface
from src.core.video_processor import VideoFrame


class EventType(Enum):
    ACTION_EVENT = "action_event"
    OBJECT_EVENT = "object_event"
    INTERACTION_EVENT = "interaction_event"
    SCENE_CHANGE = "scene_change"
    ACTIVITY_EVENT = "activity_event"
    UNKNOWN = "unknown"


@dataclass
class DetectedEvent:
    timestamp: float
    event_type: EventType
    subtype: str
    description: str
    severity: str  # low, medium, high
    confidence: float
    objects_involved: List[str]
    frame_number: int


class EventDetector:
    def __init__(self, vlm_interface: VLMInterface):
        self.vlm = vlm_interface
        self.detection_prompts = {
            "general": """Analyze this video frame and identify any significant events, activities, or objects. Look for:
1. Actions or movements (any physical activity or motion)
2. Objects and their presence (any items, entities, or elements in the scene)
3. Scene changes or transitions (camera movements, location shifts)
4. Activities or events happening (ongoing processes or events)
5. Notable elements in the scene (anything worth mentioning)

Respond in format: EVENT_TYPE|DESCRIPTION|SEVERITY|OBJECTS
If no significant events, respond: NONE

Examples:
- ACTION_EVENT|Entity moving across the scene|medium|entity_1
- OBJECT_EVENT|Object detected in the environment|low|object_1
- INTERACTION_EVENT|Multiple entities interacting|high|entity_1,entity_2
- SCENE_CHANGE|Camera movement or transition|medium|camera
- ACTIVITY_EVENT|Ongoing activity in the scene|medium|entity_1,environment

IMPORTANT: Be specific and descriptive. If you cannot provide structured format, describe what you see in natural language and I will parse it.""",
            
            "detailed": """Provide a detailed analysis of this video frame. Identify:
1. What is happening in the scene
2. Who or what is involved
3. The context or setting
4. Any notable actions or interactions
5. Objects and their purposes

Format: EVENT_TYPE|Detailed description|SEVERITY|Objects involved
If nothing notable, respond: NONE"""
        }

    def detect_events_in_frame(self, video_frame: VideoFrame) -> List[DetectedEvent]:
        """Detect events in a single frame"""
        response = self.vlm.analyze_frame(
            video_frame.frame,
            self.detection_prompts["general"]
        )
        return self._parse_event_response(
            response,
            video_frame.timestamp,
            video_frame.frame_number
        )

    def _parse_event_response(self, response: str, timestamp: float, frame_number: int) -> List[DetectedEvent]:
        """Parse VLM response into structured events"""
        events = []
        
        # If response contains "NONE" or is empty, return no events
        if "NONE" in response.upper() or not response.strip():
            return events

        # Try to parse structured format first (with | separators)
        lines = response.split('\n')
        for line in lines:
            if '|' in line:
                parts = line.split('|')
                if len(parts) >= 3:
                    event_type_str = parts[0].strip()
                    description = parts[1].strip()
                    severity = parts[2].strip().lower()
                    objects = parts[3].split(',') if len(parts) > 3 else []

                    events.append(DetectedEvent(
                        timestamp=timestamp,
                        event_type=self._classify_event_type(event_type_str),
                        subtype=self._extract_subtype(description),
                        description=description,
                        severity=severity,
                        confidence=0.8,
                        objects_involved=[obj.strip() for obj in objects],
                        frame_number=frame_number
                    ))

        # If no structured events found, try to extract from natural language
        if not events:
            events = self._extract_events_from_natural_language(response, timestamp, frame_number)
        
        return events

    def _extract_events_from_natural_language(self, response: str, timestamp: float, frame_number: int) -> List[DetectedEvent]:
        """Extract events from natural language response"""
        events = []
        response_lower = response.lower()
        
        # Generic object detection
        if any(word in response_lower for word in ['object', 'item', 'thing', 'element', 'entity']):
            events.append(DetectedEvent(
                timestamp=timestamp,
                event_type=EventType.OBJECT_EVENT,
                subtype="object_detected",
                description="Object or entity detected in the scene",
                severity="medium",
                confidence=0.7,
                objects_involved=["object"],
                frame_number=frame_number
            ))
        
        # Generic movement detection
        if any(word in response_lower for word in ['moving', 'motion', 'action', 'activity', 'movement']):
            events.append(DetectedEvent(
                timestamp=timestamp,
                event_type=EventType.ACTION_EVENT,
                subtype="movement_detected",
                description="Movement or action detected in the scene",
                severity="low",
                confidence=0.6,
                objects_involved=["entity"],
                frame_number=frame_number
            ))
        
        # Generic interaction detection
        if any(word in response_lower for word in ['interaction', 'interacting', 'together', 'between']):
            events.append(DetectedEvent(
                timestamp=timestamp,
                event_type=EventType.INTERACTION_EVENT,
                subtype="interaction_detected",
                description="Interaction between entities detected",
                severity="medium",
                confidence=0.6,
                objects_involved=["entity"],
                frame_number=frame_number
            ))
        
        # If no specific events found but response is substantial, create a general event
        if not events and len(response.strip()) > 50:
            events.append(DetectedEvent(
                timestamp=timestamp,
                event_type=EventType.OBJECT_EVENT,
                subtype="scene_analysis",
                description=f"Scene analysis: {response[:100]}...",
                severity="low",
                confidence=0.5,
                objects_involved=["scene"],
                frame_number=frame_number
            ))
        
        return events

    def _classify_event_type(self, event_text: str) -> EventType:
        """Classify event into main categories"""
        event_lower = event_text.lower()
        
        # Generic action-related keywords
        action_keywords = ['action', 'movement', 'motion', 'activity', 'performing', 'doing']
        
        # Generic object-related keywords
        object_keywords = ['object', 'item', 'thing', 'element', 'entity', 'presence']
        
        # Generic interaction-related keywords
        interaction_keywords = ['interaction', 'together', 'between', 'interacting', 'connection']
        
        # Scene change keywords
        scene_keywords = ['camera', 'pan', 'zoom', 'transition', 'scene', 'location', 'setting']
        
        # Activity keywords
        activity_keywords = ['activity', 'process', 'ongoing', 'happening', 'event']

        if any(keyword in event_lower for keyword in action_keywords):
            return EventType.ACTION_EVENT
        elif any(keyword in event_lower for keyword in interaction_keywords):
            return EventType.INTERACTION_EVENT
        elif any(keyword in event_lower for keyword in scene_keywords):
            return EventType.SCENE_CHANGE
        elif any(keyword in event_lower for keyword in activity_keywords):
            return EventType.ACTIVITY_EVENT
        elif any(keyword in event_lower for keyword in object_keywords):
            return EventType.OBJECT_EVENT
        else:
            return EventType.UNKNOWN

    def _extract_subtype(self, description: str) -> str:
        """Extract a subtype from the event description"""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['moving', 'motion', 'action']):
            return "movement"
        elif any(word in description_lower for word in ['object', 'item', 'thing']):
            return "object_detected"
        elif any(word in description_lower for word in ['interaction', 'together']):
            return "interaction"
        elif any(word in description_lower for word in ['camera', 'scene', 'transition']):
            return "scene_change"
        elif any(word in description_lower for word in ['activity', 'process']):
            return "activity"
        else:
            return "general"
