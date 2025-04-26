"""
/utils/llm_utils.py
Utility functions for interacting with the Gemini LLM API.
"""

import os
import json
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv
import google.generativeai as genai

s
load_dotenv()


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class LLMManager:
    """
    Manager class for interacting with the Gemini LLM.
    """
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        """
        Initialize the LLM Manager.
        
        Args:
            model_name: The name of the Gemini model to use
        """
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        self.history = []
    
    def query(self, 
              prompt: str, 
              system_prompt: Optional[str] = None,
              temperature: float = 0.7, 
              structured_output: Optional[Dict] = None) -> str:
        """
        Send a query to the LLM.
        
        Args:
            prompt: The user prompt to send
            system_prompt: Optional system prompt for context
            temperature: Generation temperature (0.0 to 1.0)
            structured_output: Optional schema for structured output
            
        Returns:
            The text response from the LLM
        """
        try:
            generation_config = {
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 1024,
            }
            
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
            
           
            content = []
            if system_prompt:
                content.append({"role": "system", "parts": [system_prompt]})
            
           
            for message in self.history:
                content.append(message)
                
            
            content.append({"role": "user", "parts": [prompt]})
            
            
            response = self.model.generate_content(
                content,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            
            self.history.append({"role": "user", "parts": [prompt]})
            self.history.append({"role": "model", "parts": [response.text]})
            
            return response.text
        
        except Exception as e:
            return f"Error in LLM query: {str(e)}"
    
    def query_with_json_output(self, prompt: str, json_schema: Dict, system_prompt: Optional[str] = None) -> Dict:
        """
        Query the LLM and parse the response as JSON.
        
        Args:
            prompt: The prompt to send
            json_schema: The JSON schema to follow
            system_prompt: Optional system prompt
            
        Returns:
            Parsed JSON response
        """
        schema_prompt = f"""
        You must respond with a valid JSON object that follows this schema:
        {json.dumps(json_schema, indent=2)}
        
        Only respond with the JSON object, nothing else.
        """
        
        full_prompt = f"{prompt}\n\n{schema_prompt}"
        
        try:
            response = self.query(full_prompt, system_prompt=system_prompt, temperature=0.2)
           
            return json.loads(response)
        except json.JSONDecodeError:
            
            try:
               
                    json_text = response.split("```json")[1].split("```")[0].strip()
                    return json.loads(json_text)
                elif "```" in response:
                    json_text = response.split("```")[1].split("```")[0].strip()
                    return json.loads(json_text)
                else:
                    
                    import re
                    json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
                    match = re.search(json_pattern, response)
                    if match:
                        return json.loads(match.group(0))
                    raise ValueError("Could not find valid JSON in response")
            except Exception as e:
                return {"error": f"Failed to parse JSON response: {str(e)}", "raw_response": response}
    
    def clear_history(self):
        """Clear the conversation history."""
        self.history = []
