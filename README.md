# Visual Understanding Chat Assistant

A general-purpose AI-powered application for analyzing video content, detecting events, summarizing, and enabling natural language conversations about the analyzed content. Built for flexibility and extensibility—works with any video type (meetings, sports, cooking, education, etc.).

## Features

- **Universal Video Analysis**: Works with any video content
- **Event Detection**: Identifies actions, objects, interactions, and scene changes
- **Summarization**: Generates concise, context-aware summaries
- **Conversational AI**: Multi-turn chat about video content
- **Modular, Extensible Architecture**

## Quick Start

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running (for Vision-Language Model)

### Installation
```bash
# Clone the repository and navigate to the project directory
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt

# Start Ollama and pull the required model
ollama serve
ollama pull llava:7b
```

### Running the Application
```bash
# Start the backend API server
uvicorn src.api.main:app --reload

# In a separate terminal, start the frontend
streamlit run frontend/streamlit_app.py
```

## Usage

1. Upload a video file via the Streamlit web interface
2. Click "Analyze Video" to process
3. Ask questions about the video in the chat interface (e.g., "What happened?", "Describe the main activities.", "List all objects detected.")

## Project Structure

```
Cligue/
├── README.md            # This file
├── requirements.txt     # Python dependencies
├── src/
│   ├── core/            # Video processing, VLM interface, event detection, summarization
│   ├── agents/          # Chat agent, memory management
│   ├── api/             # FastAPI backend
│   └── utils/           # Helpers, config
├── frontend/
│   └── streamlit_app.py # Streamlit frontend
└── tests/
    └── test_integration.py
```

## Architecture

### Core Components

**Video Processing (`src/core/video_processor.py`)**
- Frame extraction and validation
- Video format support and preprocessing
- Timestamp mapping and metadata handling

**VLM Interface (`src/core/vlm_interface.py`)**
- Communication with Ollama-based LLaVA model
- Enhanced prompts for better analysis quality
- Retry logic and error handling

**Event Detection (`src/core/event_detector.py`)**
- Generic event classification (actions, objects, interactions, scene changes, activities)
- Domain-agnostic detection logic
- Structured event parsing and categorization

**Summarization (`src/core/summarizer.py`)**
- LLM-powered content summarization
- Timeline generation and key highlights extraction
- Statistical analysis and event categorization

**Chat Agent (`src/agents/chat_agent.py`)**
- Multi-turn conversation management
- Context-aware responses with enhanced prompts
- Event search and filtering capabilities

**Memory Management (`src/agents/memory_manager.py`)**
- Conversation history tracking
- Context optimization and management

### API Endpoints

- `POST /upload_video` - Upload and analyze video
- `POST /chat` - Send chat messages
- `GET /health` - Health check
- `GET /status` - System status
- `GET /analysis` - Get analysis results

## Troubleshooting

### Common Issues

**Ollama Connection Issues**
```bash
# Check if Ollama is running
ollama list

# Restart Ollama service
ollama serve

# Verify model is available
ollama list | grep llava
```

**Missing Dependencies**
```bash
# Ensure virtual environment is activated
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Reinstall dependencies
pip install -r requirements.txt
```

**API/Frontend Connection Issues**
- Ensure both servers are running (API on port 8000, Streamlit on port 8501)
- Check firewall settings
- Verify Ollama is accessible at http://localhost:11434

**Video Processing Errors**
- Use supported video formats (MP4, AVI, MOV)
- Check OpenCV installation: `pip install opencv-python`
- Limit video duration (default: 2 minutes max)

### Performance Optimization

**For Faster Processing**
- Use smaller video files
- Reduce frame extraction rate in `src/core/video_processor.py`
- Ensure sufficient RAM (8GB+ recommended)

**For Better VLM Performance**
- Use GPU if available
- Ensure Ollama has sufficient resources
- Consider using a smaller model for testing

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Structure
- **Modular Design**: Each component is self-contained with clear interfaces
- **Generic Implementation**: No domain-specific logic, works with any video type
- **Extensible Architecture**: Easy to add new event types or analysis features

### Adding New Features
1. **New Event Types**: Extend `EventType` enum in `src/core/event_detector.py`
2. **Custom Analysis**: Add new methods to core modules
3. **API Endpoints**: Extend `src/api/main.py` with new routes
4. **Frontend Features**: Modify `frontend/streamlit_app.py`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes (keep code modular and generic)
4. Submit a pull request with a clear description

## License

This project is for educational and research purposes.
