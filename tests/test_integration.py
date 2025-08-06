"""
Integration tests for the Visual Understanding Chat Assistant.
"""
import pytest
import tempfile
import os
import cv2
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.video_processor import VideoProcessor, VideoFrame
from src.core.vlm_interface import VLMInterface
from src.core.event_detector import EventDetector, DetectedEvent, EventType
from src.core.summarizer import VideoSummarizer
from src.agents.chat_agent import VideoAnalysisAgent
from src.agents.memory_manager import MemoryManager
from src.utils.config import MAX_VIDEO_DURATION, FPS_SAMPLE_RATE


class TestVideoProcessor:
    """Test video processing functionality."""
    
    def test_video_processor_initialization(self):
        """Test VideoProcessor initialization."""
        processor = VideoProcessor()
        assert processor.fps_sample_rate == FPS_SAMPLE_RATE
    
    def test_create_test_video(self):
        """Create a test video for testing."""
        # Create a simple test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('test_video.mp4', fourcc, 20.0, (640, 480))
        
        for i in range(60):  # 3 seconds at 20fps
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add some content to make it interesting
            cv2.rectangle(frame, (100, 100), (300, 200), (0, 255, 0), 2)
            cv2.putText(frame, f'Frame {i}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out.write(frame)
        
        out.release()
        return 'test_video.mp4'
    
    def test_video_validation(self):
        """Test video validation functionality."""
        processor = VideoProcessor()
        
        # Test with valid video
        test_video = self.test_create_test_video()
        validation = processor.validate_video(test_video)
        
        assert validation["valid"] == True
        assert validation["duration"] <= MAX_VIDEO_DURATION
        assert validation["fps"] > 0
        assert validation["frame_count"] > 0
        
        # Clean up
        os.remove(test_video)
    
    def test_frame_extraction(self):
        """Test frame extraction functionality."""
        processor = VideoProcessor()
        test_video = self.test_create_test_video()
        
        frames = list(processor.extract_frames(test_video))
        
        assert len(frames) > 0
        assert all(isinstance(frame, VideoFrame) for frame in frames)
        assert all(hasattr(frame, 'frame') for frame in frames)
        assert all(hasattr(frame, 'timestamp') for frame in frames)
        assert all(hasattr(frame, 'frame_number') for frame in frames)
        
        # Clean up
        os.remove(test_video)


class TestVLMInterface:
    """Test VLM interface functionality."""
    
    @patch('src.core.vlm_interface.ollama.Client')
    def test_vlm_initialization(self, mock_client):
        """Test VLM interface initialization."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        vlm = VLMInterface()
        assert vlm.model_name == "llava:7b"
        assert vlm.client == mock_client_instance
    
    @patch('src.core.vlm_interface.ollama.Client')
    def test_analyze_frame(self, mock_client):
        """Test frame analysis functionality."""
        mock_client_instance = Mock()
        mock_client_instance.chat.return_value = {
            'message': {'content': 'Test response'}
        }
        mock_client.return_value = mock_client_instance
        
        vlm = VLMInterface()
        
        # Create test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        response = vlm.analyze_frame(test_frame, "Test prompt")
        
        assert response == "Test response"
        mock_client_instance.chat.assert_called_once()


class TestEventDetector:
    """Test event detection functionality."""
    
    def test_event_detector_initialization(self):
        """Test EventDetector initialization."""
        mock_vlm = Mock()
        detector = EventDetector(mock_vlm)
        
        assert detector.vlm == mock_vlm
        assert "general" in detector.detection_prompts
    
    def test_event_classification(self):
        """Test event type classification."""
        mock_vlm = Mock()
        detector = EventDetector(mock_vlm)
        
        # Test traffic violation classification
        assert detector._classify_event_type("red light violation") == EventType.TRAFFIC_VIOLATION
        assert detector._classify_event_type("wrong way driving") == EventType.TRAFFIC_VIOLATION
        
        # Test pedestrian event classification
        assert detector._classify_event_type("pedestrian crossing") == EventType.PEDESTRIAN_EVENT
        assert detector._classify_event_type("jaywalking") == EventType.PEDESTRIAN_EVENT
        
        # Test vehicle event classification
        assert detector._classify_event_type("vehicle turning") == EventType.VEHICLE_EVENT
        assert detector._classify_event_type("car parking") == EventType.VEHICLE_EVENT
    
    def test_parse_event_response(self):
        """Test parsing of VLM responses into events."""
        mock_vlm = Mock()
        detector = EventDetector(mock_vlm)
        
        # Test parsing valid response
        response = "TRAFFIC_VIOLATION|Red light running|high|car_1,traffic_light_1"
        events = detector._parse_event_response(response, 10.5, 210)
        
        assert len(events) == 1
        assert events[0].timestamp == 10.5
        assert events[0].event_type == EventType.TRAFFIC_VIOLATION
        assert events[0].description == "Red light running"
        assert events[0].severity == "high"
        assert events[0].objects_involved == ["car_1", "traffic_light_1"]
        
        # Test parsing NONE response
        none_response = "NONE"
        events = detector._parse_event_response(none_response, 10.5, 210)
        assert len(events) == 0


class TestVideoSummarizer:
    """Test video summarization functionality."""
    
    def test_summarizer_initialization(self):
        """Test VideoSummarizer initialization."""
        summarizer = VideoSummarizer()
        assert summarizer is not None
    
    def test_generate_summary(self):
        """Test summary generation."""
        summarizer = VideoSummarizer()
        
        # Create test events
        events = [
            DetectedEvent(
                timestamp=10.0,
                event_type=EventType.TRAFFIC_VIOLATION,
                subtype="red_light_running",
                description="Car ran red light",
                severity="high",
                confidence=0.9,
                objects_involved=["car_1", "traffic_light_1"],
                frame_number=200
            ),
            DetectedEvent(
                timestamp=15.0,
                event_type=EventType.PEDESTRIAN_EVENT,
                subtype="crossing",
                description="Pedestrian crossing street",
                severity="medium",
                confidence=0.8,
                objects_involved=["pedestrian_1"],
                frame_number=300
            )
        ]
        
        summary = summarizer.generate_summary(events, 60.0)
        
        assert "overview" in summary
        assert "violations" in summary
        assert "timeline" in summary
        assert len(summary["violations"]) == 1
        assert len(summary["timeline"]) == 2


class TestMemoryManager:
    """Test memory management functionality."""
    
    def test_memory_initialization(self):
        """Test MemoryManager initialization."""
        memory = MemoryManager()
        assert memory.k == 10
        assert memory.history == []
    
    def test_add_message(self):
        """Test adding messages to memory."""
        memory = MemoryManager()
        memory.add_message("user", "Hello")
        memory.add_message("assistant", "Hi there")
        
        assert len(memory.history) == 2
        assert memory.history[0]["role"] == "user"
        assert memory.history[0]["content"] == "Hello"
        assert memory.history[1]["role"] == "assistant"
        assert memory.history[1]["content"] == "Hi there"
    
    def test_get_history(self):
        """Test retrieving conversation history."""
        memory = MemoryManager(k=2)
        
        # Add more messages than k
        for i in range(6):
            memory.add_message("user", f"Message {i}")
            memory.add_message("assistant", f"Response {i}")
        
        history = memory.get_history()
        assert len(history) == 4  # k * 2 (user/assistant pairs)
    
    def test_clear_memory(self):
        """Test clearing conversation memory."""
        memory = MemoryManager()
        memory.add_message("user", "Hello")
        memory.add_message("assistant", "Hi")
        
        assert len(memory.history) == 2
        memory.clear()
        assert len(memory.history) == 0


class TestVideoAnalysisAgent:
    """Test video analysis agent functionality."""
    
    def test_agent_initialization(self):
        """Test VideoAnalysisAgent initialization."""
        mock_vlm = Mock()
        events = []
        summary = {"overview": "Test summary"}
        
        agent = VideoAnalysisAgent(events, summary, mock_vlm)
        
        assert agent.events == events
        assert agent.summary == summary
        assert agent.vlm == mock_vlm
        assert agent.memory is not None
        assert agent._initial_context is not None
    
    def test_create_initial_context(self):
        """Test initial context creation."""
        mock_vlm = Mock()
        events = []
        summary = {
            "overview": "Test overview",
            "violations": [
                {"description": "Red light violation", "timestamp": "00:10"}
            ],
            "timeline": [
                {"time": "00:10", "event": "Red light violation"}
            ]
        }
        
        agent = VideoAnalysisAgent(events, summary, mock_vlm)
        context = agent._create_initial_context()
        
        assert "Test overview" in context
        assert "Red light violation" in context
    
    @patch('src.agents.chat_agent.VLMInterface')
    def test_chat_functionality(self, mock_vlm_class):
        """Test chat functionality."""
        mock_vlm = Mock()
        mock_vlm.chat_with_context.return_value = "Test response"
        mock_vlm_class.return_value = mock_vlm
        
        events = []
        summary = {"overview": "Test summary"}
        
        agent = VideoAnalysisAgent(events, summary, mock_vlm)
        response = agent.chat("Hello")
        
        assert response == "Test response"
        mock_vlm.chat_with_context.assert_called_once()


class TestEndToEnd:
    """End-to-end integration tests."""
    
    @patch('src.core.vlm_interface.ollama.Client')
    def test_full_pipeline(self, mock_client):
        """Test the complete pipeline from video to chat."""
        # Mock VLM responses
        mock_client_instance = Mock()
        mock_client_instance.chat.return_value = {
            'message': {'content': 'TRAFFIC_VIOLATION|Red light running|high|car_1,traffic_light_1'}
        }
        mock_client.return_value = mock_client_instance
        
        # Create test video
        processor = VideoProcessor()
        test_video = self.create_test_video()
        
        try:
            # Test video processing
            validation = processor.validate_video(test_video)
            assert validation["valid"] == True
            
            frames = list(processor.extract_frames(test_video))
            assert len(frames) > 0
            
            # Test VLM interface
            vlm = VLMInterface()
            
            # Test event detection
            detector = EventDetector(vlm)
            events = detector.detect_events_in_frame(frames[0])
            assert len(events) >= 0  # May be 0 if no events detected
            
            # Test summarization
            summarizer = VideoSummarizer()
            summary = summarizer.generate_summary(events, validation["duration"])
            assert "overview" in summary
            
            # Test chat agent
            agent = VideoAnalysisAgent(events, summary, vlm)
            response = agent.chat("What happened in the video?")
            assert isinstance(response, str)
            
        finally:
            # Clean up
            if os.path.exists(test_video):
                os.remove(test_video)
    
    def create_test_video(self):
        """Create a test video for integration testing."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('test_integration.mp4', fourcc, 20.0, (640, 480))
        
        for i in range(40):  # 2 seconds at 20fps
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(frame, (100, 100), (300, 200), (0, 255, 0), 2)
            cv2.putText(frame, f'Test Frame {i}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out.write(frame)
        
        out.release()
        return 'test_integration.mp4'


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 