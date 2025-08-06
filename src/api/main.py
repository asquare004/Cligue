"""
FastAPI backend for the Visual Understanding Chat Assistant.
"""
import tempfile
import os
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

from src.core.video_processor import VideoProcessor
from src.core.vlm_interface import VLMInterface
from src.core.event_detector import EventDetector
from src.core.summarizer import VideoSummarizer
from src.agents.chat_agent import VideoAnalysisAgent
from src.api.models import UploadResponse, ChatRequest, ChatResponse, ChatMessage, StatusResponse, VideoStatistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cligue - Visual Understanding Chat Assistant")

# Global state (in-memory for hackathon)
analysis_state: Dict[str, Any] = {
    "agent": None,
    "events": [],
    "summary": {},
    "vlm_available": False,
}


@app.post("/upload_video", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """Upload and analyze video"""
    logger.info(f"Starting video upload: {file.filename}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        temp_path = tmp_file.name
        logger.info(f"Video saved to temp file: {temp_path}")

    try:
        # Initialize components
        logger.info("Initializing video processor...")
        video_processor = VideoProcessor()
        
        logger.info("Initializing VLM interface...")
        vlm_interface = VLMInterface()
        
        # Check VLM availability
        logger.info("Checking VLM availability...")
        if not vlm_interface.is_available():
            logger.error("VLM not available")
            raise HTTPException(
                status_code=503, 
                detail="VLM not available. Please ensure Ollama is running and the model is loaded."
            )
        
        logger.info("Initializing event detector...")
        event_detector = EventDetector(vlm_interface)
        
        logger.info("Initializing summarizer...")
        summarizer = VideoSummarizer(vlm_interface)

        # Validate video
        logger.info("Validating video...")
        validation = video_processor.validate_video(temp_path)
        if not validation["valid"]:
            logger.error(f"Video validation failed: {validation['error']}")
            raise HTTPException(status_code=400, detail=validation["error"])

        # Process video
        logger.info("Processing video frames...")
        events = []
        frame_count = 0
        for frame in video_processor.extract_frames(temp_path):
            try:
                frame_events = event_detector.detect_events_in_frame(frame)
                events.extend(frame_events)
                frame_count += 1
                logger.info(f"Processed frame {frame_count}, found {len(frame_events)} events")
                
                # Process more frames for better analysis
                if frame_count > 50:  # Process max 50 frames for better coverage
                    break
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {e}")
                continue

        logger.info(f"Total events detected: {len(events)}")

        # Generate summary
        logger.info("Generating summary...")
        summary = summarizer.generate_summary(events, validation["duration"])

        # Initialize and store agent and analysis
        logger.info("Initializing chat agent...")
        analysis_state["agent"] = VideoAnalysisAgent(events, summary, vlm_interface)
        analysis_state["events"] = events
        analysis_state["summary"] = summary
        analysis_state["vlm_available"] = True

        # Prepare response data
        events_by_type = summary.get("events_by_type", {})
        key_highlights = summary.get("key_highlights", [])
        statistics = summary.get("statistics", {})

        logger.info("Upload completed successfully")
        return UploadResponse(
            status="success",
            video_duration=validation["duration"],
            events_detected=len(events),
            summary=summary["overview"],
            events_by_type=events_by_type,
            key_highlights=key_highlights,
            statistics=VideoStatistics(**statistics)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during video upload: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            logger.info(f"Cleaned up temp file: {temp_path}")


@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Handle chat messages"""
    agent = analysis_state.get("agent")
    if not agent:
        raise HTTPException(status_code=400, detail="No video uploaded yet")

    try:
        response = agent.chat(message.message)
        return ChatResponse(response=response, status="success")
    except Exception as e:
        return ChatResponse(
            response=f"Error processing chat: {str(e)}", 
            status="error"
        )


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current analysis status"""
    agent = analysis_state.get("agent")
    summary = analysis_state.get("summary", {})
    vlm_available = analysis_state.get("vlm_available", False)
    
    return StatusResponse(
        video_loaded=agent is not None,
        events_count=len(analysis_state.get("events", [])),
        has_events=len(analysis_state.get("events", [])) > 0,
        vlm_available=vlm_available
    )


@app.get("/analysis")
async def get_analysis():
    """Get complete video analysis"""
    summary = analysis_state.get("summary", {})
    if not summary:
        raise HTTPException(status_code=400, detail="No video analysis available")
    
    return summary


@app.get("/events/{event_type}")
async def get_events_by_type(event_type: str):
    """Get events of a specific type"""
    agent = analysis_state.get("agent")
    if not agent:
        raise HTTPException(status_code=400, detail="No video uploaded yet")
    
    events = agent.search_events_by_type(event_type)
    return {"events": events, "type": event_type}


@app.get("/highlights")
async def get_highlights():
    """Get key highlights from the video"""
    summary = analysis_state.get("summary", {})
    if not summary:
        raise HTTPException(status_code=400, detail="No video analysis available")
    
    return {"highlights": summary.get("key_highlights", [])}


@app.get("/statistics")
async def get_statistics():
    """Get video analysis statistics"""
    summary = analysis_state.get("summary", {})
    if not summary:
        raise HTTPException(status_code=400, detail="No video analysis available")
    
    return summary.get("statistics", {})


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    vlm_interface = VLMInterface()
    return {
        "status": "healthy",
        "vlm_available": vlm_interface.is_available(),
        "timestamp": "2024-01-01T00:00:00Z"
    }


if __name__ == "__main__":
    import uvicorn
    from src.utils.config import API_HOST, API_PORT
    uvicorn.run(app, host=API_HOST, port=API_PORT)
