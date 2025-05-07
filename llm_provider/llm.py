from abc import ABC, abstractmethod
from typing import Any, Dict, Literal
import os
import json
from google import genai
import ollama


class LLMBackend(ABC):
    @abstractmethod
    def structured_generate(
        self, system_prompt: str, user_prompt: str, json_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        pass


class OllamaBackend(LLMBackend):
    def __init__(
        self, model: str = "gemma3:1b", base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.model_name = model  # for consistency with optimize_item
        self.client = ollama.Client(host=base_url)

    def structured_generate(
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
            return json.loads(response.message.content)
        except Exception as e:
            print("Error during Ollama structured prediction:", e)
            return {}

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        print("Optimizing Item")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        print(f"Sent item to {self.model_name}")
        try:
            response = self.client.chat(model=self.model_name, messages=messages)
            optimized_content = response.message.content
            print(f"Response from {self.model_name}")
            return optimized_content
        except Exception as e:
            print("Error during optimization with Ollama:", e)
            return ""


class GeminiBackend(LLMBackend):
    def __init__(self, model: str = "gemini-2.0-flash"):
        self.model = model
        self.model_name = model  # for consistency with optimize_item
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        if not self.client:
            raise ValueError("Gemini client initialization failed. Check your API key.")

    def structured_generate(
        self, system_prompt: str, user_prompt: str, json_schema: Dict[str, Any]
    ) -> Dict[str, Any]:

        response_mime_type = "application/json"
        genai_schema = genai.types.Schema().from_orm(json_schema)

        response = self.client.models.generate_content(
            model=self.model,
            contents=[user_prompt],
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.5,
                response_mime_type=response_mime_type,
                response_schema=genai_schema,
            ),
        )
        output = json.loads(response.text)
        return output

    def generate(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:

        response = self.client.models.generate_content(
            model=self.model,
            contents=[user_prompt],
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.5,
            ),
        )
        output = response.text
        return output


class LLMClient:
    def __init__(self, backend: Literal["ollama", "gemini"], **kwargs):
        if backend == "ollama":
            self.backend: LLMBackend = OllamaBackend(
                model=kwargs.get("model", "gemma3:4b"),
                base_url=kwargs.get("base_url", "http://localhost:11434"),
            )
        elif backend == "gemini":
            model = kwargs.get("model", "gemini-2.0-flash")
            self.backend: LLMBackend = GeminiBackend(model=model)
        else:
            raise ValueError("Unsupported backend. Choose from 'ollama' or 'gemini'.")

    def structured_generate(
        self, system_prompt: str, user_prompt: str, json_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self.backend.structured_generate(system_prompt, user_prompt, json_schema)

    def generate(
        self, system_prompt: str, user_prompt: str  
    ) -> Dict[str, Any]:
        return self.backend.generate(system_prompt, user_prompt)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    llm_generate = LLMClient(backend='ollama')
    # Define test prompts
    system_prompt = "You are a helpful assistant that returns funny responess."
    user_prompt = "Summarize the key points of the following job description: 'We need a Python developer with experience in APIs and data pipelines.'"
    json_schema = None
    response = llm_generate.generate(system_prompt, user_prompt)
