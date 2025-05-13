# models/ai_components/_loader.py
from .providers.gemini_provider import GeminiProvider
from .providers.local_provider import LocalLLMProvider  # Keep for future
from ..common.settings.settings_manager import SettingsManager

def load_llm():
    config = SettingsManager()
    provider = config.get('llm', 'provider', default='gemini')
    
    if provider == 'gemini':
        return GeminiProvider(config.get('llm', 'gemini_api_key'))
    elif provider == 'local':
        return LocalLLMProvider()  # Existing implementation
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")