"""Provider-specific LLM implementations"""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()


class LLMProvider(ABC):
    """Base class for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate a response from the LLM"""
        pass
    
    @abstractmethod
    def generate_streaming(self, prompt: str, system_prompt: Optional[str] = None, **kwargs):
        """Generate a streaming response (optional)"""
        pass


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider"""
    
    def __init__(self, model: str = "claude-3-7-sonnet-20250219", api_key: Optional[str] = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Install with: uv pip install anthropic")
        
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate response using Claude"""
        messages = [{"role": "user", "content": prompt}]
        
        # Build request parameters
        request_params = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.7),
            "messages": messages
        }
        
        # Only include system prompt if provided
        if system_prompt:
            request_params["system"] = system_prompt
        
        # Include seed if provided (for reproducibility)
        if "seed" in kwargs:
            request_params["seed"] = kwargs["seed"]
        
        response = self.client.messages.create(**request_params)
        
        return response.content[0].text
    
    def generate_streaming(self, prompt: str, system_prompt: Optional[str] = None, **kwargs):
        """Generate streaming response"""
        messages = [{"role": "user", "content": prompt}]
        
        # Build request parameters
        request_params = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.7),
            "messages": messages
        }
        
        # Only include system prompt if provided
        if system_prompt:
            request_params["system"] = system_prompt
        
        # Include seed if provided (for reproducibility)
        if "seed" in kwargs:
            request_params["seed"] = kwargs["seed"]
        
        with self.client.messages.stream(**request_params) as stream:
            for text in stream.text_stream:
                yield text


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider"""
    
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Install with: uv pip install openai")
        
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate response using GPT"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        create_params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.7)
        }
        
        # Include seed if provided (for reproducibility)
        if "seed" in kwargs:
            create_params["seed"] = kwargs["seed"]
        
        response = self.client.chat.completions.create(**create_params)
        
        return response.choices[0].message.content
    
    def generate_streaming(self, prompt: str, system_prompt: Optional[str] = None, **kwargs):
        """Generate streaming response"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        create_params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": True
        }
        
        # Include seed if provided (for reproducibility)
        if "seed" in kwargs:
            create_params["seed"] = kwargs["seed"]
        
        stream = self.client.chat.completions.create(**create_params)
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class OpenSourceProvider(LLMProvider):
    """Base class for open-source models (Qwen, Olmo, Llama, Kimi)"""
    
    def __init__(self, model_name: str, api_base: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize open-source provider
        
        Args:
            model_name: Name of the model
            api_base: Base URL for the API (e.g., for local server or compatible API)
            api_key: API key if required
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Install with: uv pip install openai")
        
        # Use OpenAI-compatible client for open-source models
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENSOURCE_API_KEY", "dummy"),
            base_url=api_base or os.getenv("OPENSOURCE_API_BASE")
        )
        self.model = model_name
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate response using open-source model"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        create_params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.7)
        }
        
        # Include seed if provided (for reproducibility)
        if "seed" in kwargs:
            create_params["seed"] = kwargs["seed"]
        
        response = self.client.chat.completions.create(**create_params)
        
        return response.choices[0].message.content
    
    def generate_streaming(self, prompt: str, system_prompt: Optional[str] = None, **kwargs):
        """Generate streaming response"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        create_params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": True
        }
        
        # Include seed if provided (for reproducibility)
        if "seed" in kwargs:
            create_params["seed"] = kwargs["seed"]
        
        stream = self.client.chat.completions.create(**create_params)
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


def create_provider(provider_type: str, model: str, **kwargs) -> LLMProvider:
    """
    Factory function to create a provider instance
    
    Args:
        provider_type: Type of provider ('anthropic', 'openai', 'opensource')
        model: Model name/identifier
        **kwargs: Additional provider-specific arguments
    
    Returns:
        LLMProvider instance
    """
    provider_type = provider_type.lower()
    
    if provider_type == "anthropic":
        return AnthropicProvider(model=model, api_key=kwargs.get("api_key"))
    elif provider_type == "openai":
        return OpenAIProvider(model=model, api_key=kwargs.get("api_key"))
    elif provider_type == "opensource":
        return OpenSourceProvider(
            model_name=model,
            api_base=kwargs.get("api_base"),
            api_key=kwargs.get("api_key")
        )
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")

