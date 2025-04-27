import json
from abc import ABC, abstractmethod
import os
from typing import Any, Dict, Literal
import ollama

# Conditional Gemini import
try:
    from google import genai
    
except ImportError:
    genai = None



class EvaluationBackend(ABC):
    @abstractmethod
    def evaluate(
        self, system_prompt: str, user_prompt: str, json_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        pass


class OllamaBackend(EvaluationBackend):
    def __init__(
        self, model: str = "gemma3:1b", base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.client = ollama.Client(host=base_url)

    def evaluate(
        self, system_prompt: str, user_prompt: str, json_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                format=json_schema,
                options={"temperature": 0.2},
            )
            print(response.message.content)
            return json.loads(response.message.content)
        except Exception as e:
            print("Error during Ollama evaluation:", e)
            return {}


class GeminiBackend(EvaluationBackend):
    def __init__(self, model: str = "gemini-2.0-flash"):
        if not genai:
            raise ImportError(
                "Google Generative AI SDK not found. Install with `pip install google-genai`."
            )
        self.model = model
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        if not self.client:
            raise ValueError("Gemini client initialization failed. Check your API key.")

    def evaluate(
        self, system_prompt: str, user_prompt: str, json_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        
        genai_schema = genai.types.Schema().from_orm(json_schema)

        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.5,
                response_mime_type="application/json",
                response_schema= genai_schema
                    
            )
        )
        
        return json.loads(response.text)



class ResumeEvaluator:
    def __init__(self, backend: Literal["ollama", "gemini"],**kwargs):
        if backend == "ollama":
            self.model = kwargs.get("model", "gemma3:1b")
            self.backend: EvaluationBackend = OllamaBackend(
                model=self.model,
                base_url=kwargs.get("base_url", "http://localhost:11434"),
            )
        elif backend == "gemini":
            self.model = kwargs.get("model", "gemini-2.0-flash")
            self.backend: EvaluationBackend = GeminiBackend(
                model=self.model,
            )
        else:
            raise ValueError("Unsupported backend. Choose from 'ollama' or 'gemini'.")

    def evaluate(
        self, system_prompt: str, user_prompt: str, json_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self.backend.evaluate(system_prompt, user_prompt, json_schema)
