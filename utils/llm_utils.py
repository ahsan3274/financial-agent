"""
/utils/llm_utils.py
Utility functions for interacting with the Gemini LLM API.
"""

import os
import json
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv
import google.generativeai as genai
import re

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
            Parsed JSON response or error dictionary
        """
        schema_prompt = f"""
        You must respond with a valid JSON object that follows this schema:
        {json.dumps(json_schema, indent=2)}

        Only respond with the JSON object, nothing else. Do not add explanations or markdown formatting like ```json.
        """

        full_prompt = f"{prompt}\n\n{schema_prompt}"

        try:
            # First attempt: Assume the LLM returns *only* valid JSON
            response = self.query(full_prompt, system_prompt=system_prompt, temperature=0.1) # Lower temp for stricter formats
            return json.loads(response)
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON from potentially noisy response
            try:
                # Look for JSON between triple backticks (common mistake by LLMs)
                if "```json" in response:
                    # Extract content after ```json and before the next ```
                    json_text = response.split("```json", 1)[1].split("```", 1)[0].strip()
                    return json.loads(json_text)
                elif "```" in response:
                     # Extract content between the first pair of ```
                    json_text = response.split("```", 1)[1].split("```", 1)[0].strip()
                     # Sometimes the LLM might still put 'json' after the first backticks
                    if json_text.lower().startswith('json'):
                         json_text = json_text[4:].strip()
                    return json.loads(json_text)
                else:
                    # Last resort: Find the first opening curly brace and try to parse from there
                    # This is less reliable but can sometimes salvage responses
                    first_brace = response.find('{')
                    last_brace = response.rfind('}')
                    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                        potential_json = response[first_brace:last_brace+1]
                        try:
                            # Try parsing this substring
                            return json.loads(potential_json)
                        except json.JSONDecodeError:
                            # If substring parsing fails, fall through to raise error
                            pass
                    # If no backticks and substring parsing failed, raise error to be caught below
                    raise ValueError("Could not find valid JSON structure (backticks or braces) in response")
            except Exception as e:
                # Catch errors from splitting, indexing, or json.loads in the extraction block
                error_message = f"Failed to parse JSON response after initial failure. Extraction Error: {str(e)}"
                # print(f"DEBUG: Raw response that failed parsing:\n---\n{response}\n---") # Uncomment for debugging
                return {"error": error_message, "raw_response": response}
        except Exception as e:
            # Catch any other unexpected errors during the initial self.query or json.loads attempt
             error_message = f"An unexpected error occurred during JSON query: {str(e)}"
             return {"error": error_message, "raw_response": "Response not available due to error"}
    def clear_history(self):
        """Clear the conversation history."""
        self.history = []
