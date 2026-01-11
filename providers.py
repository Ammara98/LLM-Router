from abc import ABC, abstractmethod
from typing import Optional
import os
import json
from pydantic import BaseModel


class LLMProvider(ABC):

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        pass
    
    def generate_json(self, prompt: str, system_prompt: str, model_class: type[BaseModel]) -> BaseModel:
        full_prompt = f"""{system_prompt}

        Query: {prompt}

        Respond with a JSON object with these fields:
        - intent: string (one of: "faq", "order_status", "unclear")
        - confidence: number (0.0 to 1.0)
        - entities: object (e.g. {{"order_id": "ORD-12345"}})
        - reasoning: string (brief explanation)

        Example response:
        {{"intent": "faq", "confidence": 0.95, "entities": {{}}, "reasoning": "Query asks about store hours"}}

        IMPORTANT: Return ONLY the JSON object with intent/confidence/entities/reasoning. No schema, no markdown, no explanations."""
        
        response = self.generate(full_prompt)
        
        response = response.strip()

        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        
        # HuggingFace often includes the schema - extract only the data part
        if '"$defs"' in response or '"properties"' in response:
            # Schema is present, find the actual data after it
            parts = response.split('}, {')
            if len(parts) >= 2:
                # Take the last part (actual data)
                response = '{' + parts[-1]
            else:
                import re
                json_objects = re.findall(r'\{[^{}]*"intent"[^{}]*\}', response)
                if json_objects:
                    response = json_objects[-1]
        
        if '{' in response:
            response = response[response.index('{'):]

        if '}' in response:
            response = response[:response.rindex('}')+1]
        
        response = response.strip()
        
        try:
            data = json.loads(response)
            if 'intent' in data and isinstance(data['intent'], str):
                data['intent'] = data['intent'].lower()
            return model_class(**data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse JSON: {e}\nResponse: {response[:200]}")


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY required")
        
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel("gemini-2.5-flash")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        full = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        return self.client.generate_content(full).text
    
    def get_model_name(self) -> str:
        return "gemini-flash"


class GroqProvider(LLMProvider):

    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY required")
        
        from groq import Groq
        self.client = Groq(api_key=api_key)
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.1
        )
        return response.choices[0].message.content
    
    def get_model_name(self) -> str:
        return "groq-llama"


# class HuggingFaceProvider(LLMProvider):
#     def __init__(self, api_key: Optional[str] = None):
#         api_key = api_key or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
#         if not api_key:
#             raise ValueError("HUGGINGFACE_API_KEY or HF_TOKEN required")
        
#         from huggingface_hub import InferenceClient
#         self.client = InferenceClient(api_key=api_key)
    
#     def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
#         messages = []
#         if system_prompt:
#             messages.append({"role": "system", "content": system_prompt})
#         messages.append({"role": "user", "content": prompt})
        
#         completion = self.client.chat.completions.create(
#             model="allenai/Olmo-3.1-32B-Think:publicai",
#             messages=messages,
#             max_tokens=1024,
#             temperature=0.1
#         )
#         return completion.choices[0].message.content
    
#     def get_model_name(self) -> str:
#         return "hf-olmo"


class OllamaProvider(LLMProvider):

    def __init__(self, model_name: str = "qwen2.5:3b", base_url: Optional[str] = None):
        try:
            import ollama
        except ImportError:
            raise ValueError(
                "ollama package not installed. Install with: pip install ollama"
            )
        
        self.model_name = model_name
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        if base_url or os.getenv("OLLAMA_BASE_URL"):
            self.client = ollama.Client(host=self.base_url)
        else:
            self.client = ollama.Client()
        
        try:
            self.client.list()
        except Exception as e:
            raise ValueError(
                f"Cannot connect to Ollama server at {self.base_url}. "
                f"Make sure Ollama is running (ollama serve). Error: {e}"
            )
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": 0.1,
                    "num_predict": 1024,
                }
            )
            return response['message']['content']
        except Exception as e:
            raise ValueError(
                f"Ollama generation failed for model '{self.model_name}'. "
                f"Make sure the model is installed (ollama pull {self.model_name}). "
                f"Error: {e}"
            )
    
    def get_model_name(self) -> str:
        """Return model name for display"""
        return f"ollama-{self.model_name.replace(':', '-')}"