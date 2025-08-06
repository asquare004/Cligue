"""
Summarization module for creating summaries of detected events.
Uses open-source LLM for better summarization.
"""
from typing import List, Dict, Any
from src.core.event_detector import DetectedEvent, EventType
from src.utils.helpers import format_timestamp
import ollama


class VideoSummarizer:
    def __init__(self, vlm_interface=None):
        self.vlm = vlm_interface
        # Use a smaller, faster model for summarization
        self.summary_model = "llama2:7b"  # Can be changed to other models like "mistral:7b"
        
    def generate_summary(self, events: List[DetectedEvent], video_duration: float) -> Dict[str, Any]:
        """Generate comprehensive video summary using LLM"""
        summary = {
            "overview": self._generate_overview(events, video_duration),
            "events_by_type": self._categorize_events(events),
            "timeline": self._create_timeline(events),
            "key_highlights": self._generate_highlights(events),
            "statistics": self._generate_statistics(events, video_duration)
        }
        return summary

    def _generate_overview(self, events: List[DetectedEvent], duration: float) -> str:
        """Generate high-level overview using LLM"""
        if not events:
            return f"Video Analysis Summary ({format_timestamp(duration)}s):\nNo significant events detected in this video."
        
        # Prepare event data for LLM
        event_descriptions = []
        for event in events:
            event_descriptions.append(f"- {format_timestamp(event.timestamp)}: {event.description} ({event.event_type.value})")
        
        event_text = "\n".join(event_descriptions)
        
        # Create prompt for LLM
        prompt = f"""Create a concise, engaging summary of this video based on the following events:

Video Duration: {format_timestamp(duration)}s
Events Detected:
{event_text}

Please provide a natural, conversational summary that:
1. Describes what happened in the video
2. Highlights the most important events
3. Mentions the key people, objects, or activities involved
4. Gives a sense of the overall flow and context

Keep the summary engaging and informative, as if explaining to someone who hasn't seen the video."""

        try:
            # Use Ollama for summarization
            response = ollama.chat(
                model=self.summary_model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response['message']['content']
        except Exception as e:
            # Fallback to basic summary if LLM fails
            return self._generate_basic_overview(events, duration)

    def _generate_basic_overview(self, events: List[DetectedEvent], duration: float) -> str:
        """Generate basic overview without LLM"""
        total_events = len(events)
        overview = f"Video Analysis Summary ({format_timestamp(duration)}s):\n"
        overview += f"- Total events detected: {total_events}\n"
        
        if events:
            overview += "- Key events:\n"
            for event in events[:5]:  # Top 5 events
                overview += f"  • {event.description} at {format_timestamp(event.timestamp)}\n"
        
        return overview

    def _categorize_events(self, events: List[DetectedEvent]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize events by type"""
        categorized = {}
        
        for event in events:
            event_type = event.event_type.value
            if event_type not in categorized:
                categorized[event_type] = []
            
            categorized[event_type].append({
                "timestamp": format_timestamp(event.timestamp),
                "type": event.event_type.value,
                "description": event.description,
                "severity": event.severity,
                "objects": event.objects_involved
            })
        
        return categorized

    def _create_timeline(self, events: List[DetectedEvent]) -> List[Dict[str, Any]]:
        """Create chronological timeline"""
        timeline = []
        for event in sorted(events, key=lambda x: x.timestamp):
            timeline.append({
                "time": format_timestamp(event.timestamp),
                "event": event.description,
                "type": event.event_type.value,
                "severity": event.severity
            })
        return timeline

    def _generate_highlights(self, events: List[DetectedEvent]) -> List[str]:
        """Generate key highlights using LLM"""
        if not events:
            return ["No significant events detected"]
        
        # Get high-severity events
        high_severity_events = [e for e in events if e.severity == "high"]
        if not high_severity_events:
            high_severity_events = events[:3]  # Take first 3 if no high severity
        
        event_descriptions = [f"- {e.description} at {format_timestamp(e.timestamp)}" for e in high_severity_events]
        events_text = "\n".join(event_descriptions)
        
        prompt = f"""Based on these key events from a video, create 3-5 engaging highlights:

{events_text}

Please provide highlights that:
1. Are concise and interesting
2. Capture the most important moments
3. Give context about what happened
4. Are written in an engaging way

Format as a simple list of highlights."""

        try:
            response = ollama.chat(
                model=self.summary_model,
                messages=[{"role": "user", "content": prompt}]
            )
            # Parse response into list
            highlights_text = response['message']['content']
            highlights = [line.strip().lstrip('- ').lstrip('• ').lstrip('* ') 
                         for line in highlights_text.split('\n') 
                         if line.strip() and not line.strip().startswith('#')]
            return highlights[:5]  # Limit to 5 highlights
        except Exception as e:
            # Fallback to basic highlights
            return [f"{e.description} at {format_timestamp(e.timestamp)}" for e in high_severity_events[:3]]

    def _generate_statistics(self, events: List[DetectedEvent], duration: float) -> Dict[str, Any]:
        """Generate video statistics"""
        if not events:
            return {
                "total_events": 0,
                "events_per_minute": 0,
                "event_types": {},
                "severity_distribution": {},
                "duration_minutes": round(duration / 60, 2)
            }
        
        # Count events by type
        event_types = {}
        severity_dist = {"low": 0, "medium": 0, "high": 0}
        
        for event in events:
            event_type = event.event_type.value
            event_types[event_type] = event_types.get(event_type, 0) + 1
            severity_dist[event.severity] = severity_dist.get(event.severity, 0) + 1
        
        events_per_minute = len(events) / (duration / 60) if duration > 0 else 0
        
        return {
            "total_events": len(events),
            "events_per_minute": round(events_per_minute, 2),
            "event_types": event_types,
            "severity_distribution": severity_dist,
            "duration_minutes": round(duration / 60, 2)
        }
