# models/ai_components/manager/ai_model_manager.py
from ..providers import GeminiProvider
from common.singleton import Singleton

class AIModelManager(metaclass=Singleton):
    def __init__(self):
        self._load_config()
        
    def _load_config(self):
        from config.settings_manager import SettingsManager
        config = SettingsManager()
        self.provider = config.get('ai', 'provider', 'gemini')
        self.api_key = config.get_secret('ai_api_key')
        
        self._init_provider()
    
    def _init_provider(self):
        if self.provider == 'gemini':
            self.ai_provider = GeminiProvider(self.api_key)
        elif self.provider == 'openai':
            # 향후 확장을 위한 구조
            pass
            
    def get_agent(self, agent_type: str):
        return {
            'labeling': LabelingAgent(self.ai_provider),
            'validation': ValidationAgent(self.ai_provider)
        }[agent_type]# models/ai_components/manager/ai_model_manager.py
        from ..providers import GeminiProvider
        from common.singleton import Singleton
        
        class AIModelManager(metaclass=Singleton):
            def __init__(self):
                self._load_config()
                
            def _load_config(self):
                from config.settings_manager import SettingsManager
                config = SettingsManager()
                self.provider = config.get('ai', 'provider', 'gemini')
                self.api_key = config.get_secret('ai_api_key')
                
                self._init_provider()
            
            def _init_provider(self):
                if self.provider == 'gemini':
                    self.ai_provider = GeminiProvider(self.api_key)
                elif self.provider == 'openai':
                    # 향후 확장을 위한 구조
                    pass
                    
            def get_agent(self, agent_type: str):
                return {
                    'labeling': LabelingAgent(self.ai_provider),
                    'validation': ValidationAgent(self.ai_provider)
                }[agent_type]