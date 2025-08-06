"""
Conversational agent for analyzing video events and summaries.
Generic implementation for any type of video content.
"""
from typing import List, Dict, Any
from src.core.event_detector import DetectedEvent
from src.utils.helpers import format_timestamp
from src.agents.memory_manager import MemoryManager
from src.core.vlm_interface import VLMInterface


class VideoAnalysisAgent:
    def __init__(self, events: List[DetectedEvent], summary: Dict[str, Any], vlm_interface: VLMInterface):
        self.events = events
        self.summary = summary
        self.memory = MemoryManager()
        self.vlm = vlm_interface
        self._initial_context = self._create_initial_context()
        self.memory.add_message("system", self._initial_context)

    def _create_initial_context(self) -> str:
        """Creates the initial system prompt with a summary of the video analysis."""
        context = """You are an intelligent video analysis assistant. You have analyzed a video and detected various events, objects, and activities. Your role is to help users understand the video content through natural conversation.

ANALYSIS CONTEXT:
"""
        
        # Add video overview
        context += "VIDEO OVERVIEW:\n"
        context += self.summary.get("overview", "No summary available.")
        context += "\n\n"
        
        # Add statistics
        stats = self.summary.get("statistics", {})
        if stats:
            context += f"VIDEO STATISTICS:\n"
            context += f"- Total events detected: {stats.get('total_events', 0)}\n"
            context += f"- Events per minute: {stats.get('events_per_minute', 0):.1f}\n"
            context += f"- Video duration: {stats.get('duration_minutes', 0):.1f} minutes\n"
            context += "\n"
        
        # Add key highlights
        context += "KEY HIGHLIGHTS:\n"
        context += self._format_highlights_for_prompt()
        context += "\n\n"
        
        # Add timeline
        context += "TIMELINE OF EVENTS:\n"
        context += self._format_timeline_for_prompt()
        context += "\n\n"
        
        # Add events by type
        context += "EVENTS BY CATEGORY:\n"
        context += self._format_events_by_type_for_prompt()
        context += "\n\n"
        
        # Add detailed event information
        if self.events:
            context += "DETAILED EVENT ANALYSIS:\n"
            for i, event in enumerate(self.events[:15], 1):  # Show first 15 events
                context += f"{i}. {event.description} (Type: {event.event_type.value}, Severity: {event.severity}, Time: {format_timestamp(event.timestamp)})\n"
            context += "\n"
        
        context += """RESPONSE GUIDELINES:
- Be specific and reference actual events, timestamps, and details from the analysis
- Provide clear, concise explanations
- If asked about something not in the analysis, acknowledge this clearly
- Use natural, conversational language
- When appropriate, suggest related questions or topics
- Focus on what is actually visible and detected in the video
- Be helpful and informative while staying accurate to the analysis

CAPABILITIES:
- Describe what happened in the video
- Explain specific events or moments with timestamps
- Identify objects, entities, or activities detected
- Analyze patterns or behaviors
- Compare different events or time periods
- Identify the most significant moments
- Answer questions about video content
- Provide context and explanations

Remember: Always base your responses on the actual analysis data provided. If you're uncertain about something, say so clearly."""
        
        return context

    def chat(self, user_input: str) -> str:
        """Handle user chat input with context from memory."""
        self.memory.add_message("user", user_input)
        messages = self.memory.get_history()
        
        # Get response from VLM
        response = self.vlm.chat_with_context(messages)
        
        self.memory.add_message("assistant", response)
        return response

    def _format_highlights_for_prompt(self) -> str:
        """Formats the highlights list for the initial system prompt."""
        highlights = self.summary.get("key_highlights", [])
        if not highlights:
            return "No highlights available."
        
        formatted_list = []
        for i, highlight in enumerate(highlights, 1):
            formatted_list.append(f"{i}. {highlight}")
        return "\n".join(formatted_list)

    def _format_timeline_for_prompt(self) -> str:
        """Formats the timeline for the initial system prompt."""
        timeline = self.summary.get("timeline", [])
        if not timeline:
            return "No timeline available."

        formatted_list = []
        for item in timeline:
            formatted_list.append(f"- {item['time']}: {item['event']} ({item['type']})")
        return "\n".join(formatted_list)

    def _format_events_by_type_for_prompt(self) -> str:
        """Formats the events by type for the initial system prompt."""
        events_by_type = self.summary.get("events_by_type", {})
        if not events_by_type:
            return "No events categorized."
        
        formatted_list = []
        for event_type, events in events_by_type.items():
            formatted_list.append(f"{event_type.replace('_', ' ').title()}:")
            for event in events[:5]:  # Show first 5 events per type
                formatted_list.append(f"  • {event['description']} at {event['timestamp']}")
        return "\n".join(formatted_list)

    def get_video_statistics(self) -> Dict[str, Any]:
        """Get video analysis statistics."""
        return self.summary.get("statistics", {})

    def search_events_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        """Search for events of a specific type."""
        events_by_type = self.summary.get("events_by_type", {})
        return events_by_type.get(event_type, [])

    def search_events_by_time_range(self, start_time: float, end_time: float) -> List[Dict[str, Any]]:
        """Search for events within a specific time range."""
        matching_events = []
        for event in self.events:
            if start_time <= event.timestamp <= end_time:
                matching_events.append({
                    "timestamp": format_timestamp(event.timestamp),
                    "type": event.event_type.value,
                    "description": event.description,
                    "severity": event.severity,
                    "objects": event.objects_involved
                })
        return matching_events

    def get_high_severity_events(self) -> List[Dict[str, Any]]:
        """Get events with high severity."""
        high_severity = []
        for event in self.events:
            if event.severity.lower() == "high":
                high_severity.append({
                    "timestamp": format_timestamp(event.timestamp),
                    "type": event.event_type.value,
                    "description": event.description,
                    "severity": event.severity,
                    "objects": event.objects_involved
                })
        return high_severity
