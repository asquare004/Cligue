"""
VLM Interface for interacting with the Vision Language Model.
This implementation uses Ollama for CPU-based inference.
"""
import ollama
from typing import List, Dict, Optional
import numpy as np
import time
from src.utils.config import VLM_MODEL, VLM_API_BASE
from src.utils.helpers import frame_to_base64


class VLMInterface:
    def __init__(self, model_name: str = VLM_MODEL):
        self.model_name = model_name
        self.max_retries = 3
        self.retry_delay = 1.0
        try:
            self.client = ollama.Client(host=VLM_API_BASE)
            # Test connection
            self.client.list()
        except Exception as e:
            print(f"Warning: Could not connect to Ollama at {VLM_API_BASE}")
            print(f"Error: {e}")
            print("Please ensure Ollama is running and the model is available")
            self.client = None

    def chat_with_context(self, messages: List[Dict[str, str]]) -> str:
        """
        Sends a chat history to the VLM for a response.
        The last message in the list should be the user's current query.
        """
        if not self.client:
            return "Error: VLM not available. Please check Ollama connection."
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    options={
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 1024
                    }
                )
                return response['message']['content']
            except Exception as e:
                print(f"Error during VLM chat (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return f"Error communicating with the VLM: {str(e)}"

    def analyze_frame(self, frame: np.ndarray, prompt: str) -> str:
        """Analyze single frame with custom prompt (for initial analysis)."""
        if not self.client:
            return "Error: VLM not available. Please check Ollama connection."
        
        # Enhance the prompt for better analysis
        enhanced_prompt = f"""Please provide a detailed and accurate analysis of this image. 

{prompt}

Guidelines:
- Be specific and descriptive
- Focus on what is actually visible in the image
- Use clear, concise language
- If uncertain about something, acknowledge the uncertainty
- Provide structured responses when possible

Please analyze the image and respond:"""
        
        for attempt in range(self.max_retries):
            try:
                b64_image = frame_to_base64(frame)
                response = self.client.chat(
                    model=self.model_name,
                    messages=[{
                        'role': 'user',
                        'content': enhanced_prompt,
                        'images': [b64_image]
                    }],
                    options={
                        "temperature": 0.3,  # Lower temperature for more consistent analysis
                        "top_p": 0.8,
                        "num_predict": 512
                    }
                )
                return response['message']['content']
            except Exception as e:
                print(f"Error analyzing frame (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return f"Error analyzing frame: {str(e)}"

    def is_available(self) -> bool:
        """Check if the VLM is available and ready."""
        if not self.client:
            return False
        
        try:
            # Test with a simple request
            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                options={"num_predict": 10}
            )
            return True
        except Exception:
            return False

    def get_model_info(self) -> Optional[Dict]:
        """Get information about the loaded model."""
        if not self.client:
            return None
        
        try:
            models = self.client.list()
            for model in models['models']:
                if model['name'] == self.model_name:
                    return model
            return None
        except Exception:
            return None
