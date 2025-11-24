"""Unified LLM client interface"""
from typing import Optional, Dict, Any
from .providers import LLMProvider, create_provider


class LLMClient:
    """Unified interface for LLM interactions"""
    
    def __init__(self, provider_type: str, model: str, **kwargs):
        """
        Initialize LLM client
        
        Args:
            provider_type: Type of provider ('anthropic', 'openai', 'opensource')
            model: Model name/identifier
            **kwargs: Additional provider-specific arguments
        """
        self.provider = create_provider(provider_type, model, **kwargs)
        self.provider_type = provider_type
        self.model = model
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Generate a response from the LLM
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)
        
        Returns:
            Generated text response
        """
        return self.provider.generate(prompt, system_prompt, **kwargs)
    
    def generate_streaming(self, prompt: str, system_prompt: Optional[str] = None, **kwargs):
        """
        Generate a streaming response
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            **kwargs: Additional generation parameters
        
        Yields:
            Text chunks as they are generated
        """
        yield from self.provider.generate_streaming(prompt, system_prompt, **kwargs)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LLMClient":
        """
        Create LLM client from configuration dictionary
        
        Args:
            config: Configuration dict with 'provider', 'model', and optional 'api_key', 'api_base'
        
        Returns:
            LLMClient instance
        """
        return cls(
            provider_type=config["provider"],
            model=config["model"],
            api_key=config.get("api_key"),
            api_base=config.get("api_base")
        )

