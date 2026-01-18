"""
Configuration for better, more powerful models for the Hindi RAG system
"""

# Better model configurations for improved responses
BETTER_MODEL_CONFIGS = {
    # Open-source models that work well for Hindi/multilingual tasks
    "huggingface": {
        # IndicBERT or mBERT for better Hindi understanding
        "indic_transformers": {
            "model": "ai4bharat/indic-bert",
            "description": "BERT model trained specifically for Indian languages including Hindi"
        },
        # Multilingual models that work well for Hindi
        "multilingual_mistral": {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "description": "Powerful multilingual model with 7B parameters, good for Hindi"
        },
        "llama_3b_multilingual": {
            "model": "unsloth/Llama-3.2-3B-Instruct",
            "description": "Efficient 3B parameter model with good multilingual capabilities"
        },
        "xglm_4_5b": {
            "model": "facebook/xglm-4.5B",
            "description": "Multilingual model with 4.5B parameters, good for generation"
        },
        "mt5_xxl": {
            "model": "google/mt5-xxl",
            "description": "Massive multilingual T5 model with 13B parameters, excellent for generation"
        },
        "bloom_7b1": {
            "model": "bigscience/bloom-7b1",
            "description": "BigScience model with 7.1B parameters, good for multilingual tasks"
        }
    },
    
    # Commercial models for better performance
    "openai": {
        "gpt_4_turbo": {
            "model": "gpt-4-turbo-preview",
            "description": "Most capable OpenAI model, excellent for complex reasoning"
        },
        "gpt_4": {
            "model": "gpt-4",
            "description": "Highly capable model with strong reasoning abilities"
        }
    },
    
    "anthropic": {
        "claude_3_opus": {
            "model": "claude-3-opus-20240229",
            "description": "Anthropic's most capable model, excellent for complex tasks"
        },
        "claude_3_sonnet": {
            "model": "claude-3-sonnet-20240229",
            "description": "Balanced model with strong reasoning capabilities"
        },
        "claude_3_haiku": {
            "model": "claude-3-haiku-20240307",
            "description": "Fast and cost-effective model"
        }
    },
    
    "google": {
        "gemini_1_5_pro": {
            "model": "gemini-1.5-pro-latest",
            "description": "Google's most advanced multimodal model"
        },
        "gemini_pro": {
            "model": "gemini-pro",
            "description": "Google's powerful text generation model"
        }
    },
    
    "ollama": {
        # Models that can be run locally with good performance
        "llama3_70b": {
            "model": "llama3:70b",
            "description": "Llama 3 with 70B parameters for maximum capability"
        },
        "mixtral_8x7b": {
            "model": "mixtral:8x7b",
            "description": "Mixtral MoE model with 47B parameters (effective 13B)"
        },
        "mistral_large": {
            "model": "mistral-large",
            "description": "Mistral's largest model with strong multilingual capabilities"
        }
    },
    
    "local": {
        # GGUF models that can be downloaded and run locally
        "llama3_70b_instruct": {
            "model": "TheBloke/Llama-3-70B-Instruct-GGUF/llama3-70b-instruct.Q4_K_M.gguf",
            "model_type": "llama",
            "description": "Llama 3 70B parameters in quantized format for local inference"
        },
        "mixtral_8x7b_instruct": {
            "model": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
            "model_type": "mixtral",
            "description": "Mixtral 8x7B model with 47B total parameters (effective 13B active)"
        },
        "mistral_7b_instruct": {
            "model": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            "model_type": "mistral",
            "description": "Mistral 7B with instruction tuning"
        }
    }
}

def get_better_model_config(provider, model_name=None):
    """
    Get configuration for a better model
    
    Args:
        provider (str): The LLM provider (e.g., 'huggingface', 'openai', 'anthropic')
        model_name (str, optional): Specific model name to get config for
    
    Returns:
        dict: Model configuration or list of available models if model_name not specified
    """
    if provider not in BETTER_MODEL_CONFIGS:
        raise ValueError(f"Provider {provider} not in better model configs")
    
    provider_configs = BETTER_MODEL_CONFIGS[provider]
    
    if model_name:
        if model_name not in provider_configs:
            raise ValueError(f"Model {model_name} not available for provider {provider}")
        return provider_configs[model_name]
    else:
        return provider_configs

def suggest_best_models_for_hindi():
    """
    Suggest the best models for Hindi language processing
    """
    suggestions = {
        "commercial_top_tier": [
            "anthropic.claude_3_opus",
            "google.gemini_1_5_pro", 
            "openai.gpt_4_turbo"
        ],
        "commercial_balanced": [
            "anthropic.claude_3_sonnet",
            "google.gemini_pro",
            "openai.gpt_4"
        ],
        "open_source_powerful": [
            "huggingface.mt5_xxl",
            "huggingface.bloom_7b1",
            "huggingface.multilingual_mistral"
        ],
        "open_source_efficient": [
            "huggingface.llama_3b_multilingual",
            "huggingface.xglm_4_5b"
        ],
        "local_large": [
            "local.llama3_70b_instruct",
            "local.mixtral_8x7b_instruct"
        ],
        "local_balanced": [
            "local.mistral_7b_instruct",
            "ollama.llama3_70b",
            "ollama.mixtral_8x7b"
        ]
    }
    return suggestions

def update_env_with_better_config(env_file_path=".env", provider="huggingface", model_key="multilingual_mistral"):
    """
    Update the .env file with better model configuration
    """
    import os
    from pathlib import Path
    
    env_path = Path(env_file_path)
    
    # Get the better model config
    config = get_better_model_config(provider, model_key)
    
    # Read existing .env content
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        content = ""
    
    # Update the relevant lines
    lines = content.split('\n')
    updated_lines = []
    
    # Flags to track if we've updated certain values
    provider_updated = False
    model_updated = False
    
    for line in lines:
        if line.startswith('LLM_PROVIDER='):
            updated_lines.append(f'LLM_PROVIDER={provider}')
            provider_updated = True
        elif line.startswith('HUGGINGFACE_MODEL=') and provider == 'huggingface':
            updated_lines.append(f'HUGGINGFACE_MODEL={config["model"]}')
            model_updated = True
        elif line.startswith('OPENAI_MODEL=') and provider == 'openai':
            updated_lines.append(f'OPENAI_MODEL={config["model"]}')
            model_updated = True
        elif line.startswith('ANTHROPIC_MODEL=') and provider == 'anthropic':
            updated_lines.append(f'ANTHROPIC_MODEL={config["model"]}')
            model_updated = True
        elif line.startswith('GOOGLE_MODEL=') and provider == 'google':
            updated_lines.append(f'GOOGLE_MODEL={config["model"]}')
            model_updated = True
        elif line.startswith('OLLAMA_MODEL=') and provider == 'ollama':
            updated_lines.append(f'OLLAMA_MODEL={config["model"]}')
            model_updated = True
        elif line.startswith('LOCAL_MODEL_PATH=') and provider == 'local':
            updated_lines.append(f'LOCAL_MODEL_PATH={config["model"]}')
            model_updated = True
        elif line.startswith('LOCAL_MODEL_TYPE=') and provider == 'local' and 'model_type' in config:
            updated_lines.append(f'LOCAL_MODEL_TYPE={config["model_type"]}')
            model_updated = True
        else:
            updated_lines.append(line)
    
    # If we didn't find and update the provider, add it
    if not provider_updated:
        updated_lines.append(f'LLM_PROVIDER={provider}')
    
    # If we didn't find and update the model, add it
    if not model_updated:
        if provider == 'huggingface':
            updated_lines.append(f'HUGGINGFACE_MODEL={config["model"]}')
        elif provider == 'local' and 'model_type' in config:
            updated_lines.append(f'LOCAL_MODEL_PATH={config["model"]}')
            updated_lines.append(f'LOCAL_MODEL_TYPE={config["model_type"]}')
        else:
            model_var = f'{provider.upper()}_MODEL'
            updated_lines.append(f'{model_var}={config["model"]}')
    
    # Write back to file
    with open(env_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(updated_lines))
    
    print(f"Updated {env_file_path} with better model configuration:")
    print(f"  Provider: {provider}")
    print(f"  Model: {config['model']}")
    print(f"  Description: {config['description']}")
    
    return config

if __name__ == "__main__":
    print("Better Model Configurations for Hindi RAG System")
    print("=" * 50)
    
    # Show available models
    for provider, configs in BETTER_MODEL_CONFIGS.items():
        print(f"\n{provider.upper()} MODELS:")
        for model_key, config in configs.items():
            print(f"  - {model_key}: {config['description']}")
    
    print("\n" + "=" * 50)
    print("Model Suggestions for Hindi:")
    
    suggestions = suggest_best_models_for_hindi()
    for category, models in suggestions.items():
        print(f"\n{category.replace('_', ' ').upper()}:")
        for model in models:
            print(f"  - {model}")