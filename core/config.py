"""
Configuration management for CodeGen CLI
"""

import os
from typing import Dict, Any, Optional

# Default configuration
DEFAULT_CONFIG = {
    'default_model': 'gpt-3.5-turbo',
    'timeout': 30,
    'max_retries': 3,
    'verbose': False,
    'debug': False,
    'auto_save': False,
    'max_file_size': 1024 * 1024,  # 1MB
    'supported_extensions': ['.py', '.js', '.ts', '.html', '.css', '.json', '.md', '.txt'],
    'sandbox_timeout': 10,
    'max_fix_attempts': 3
}

# Available AI models
AVAILABLE_MODELS = {
    'openai': ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo'],
    'google': ['gemini-1.5-pro-latest', 'gemini-pro'],
}

def get_default_model() -> str:
    """Get the default AI model"""
    # Check if API keys are available and return appropriate default
    if os.getenv('OPENAI_API_KEY'):
        return 'gpt-3.5-turbo'
    elif os.getenv('GOOGLE_API_KEY'):
        return 'gemini-1.5-pro-latest'
    else:
        return 'mock'  # Fallback for testing

def get_config_value(key: str, default: Any = None) -> Any:
    """Get a configuration value"""
    return DEFAULT_CONFIG.get(key, default)

def get_all_available_models() -> list:
    """Get all available AI models"""
    all_models = []
    for provider_models in AVAILABLE_MODELS.values():
        all_models.extend(provider_models)
    return all_models

def is_model_available(model: str) -> bool:
    """Check if a model is available"""
    if model == 'mock':
        return True
    
    if model in AVAILABLE_MODELS['openai']:
        return bool(os.getenv('OPENAI_API_KEY'))
    elif model in AVAILABLE_MODELS['google']:
        return bool(os.getenv('GOOGLE_API_KEY'))
    
    return False

def get_api_key(provider: str) -> Optional[str]:
    """Get API key for a provider"""
    key_mapping = {
        'openai': 'OPENAI_API_KEY',
        'google': 'GOOGLE_API_KEY',
        'gemini': 'GOOGLE_API_KEY'
    }
    
    env_var = key_mapping.get(provider.lower())
    if env_var:
        return os.getenv(env_var)
    
    return None

def validate_config() -> Dict[str, bool]:
    """Validate current configuration"""
    validation = {
        'openai_key': bool(os.getenv('OPENAI_API_KEY')),
        'google_key': bool(os.getenv('GOOGLE_API_KEY')),
        'default_model_available': is_model_available(get_default_model())
    }
    
    return validation
