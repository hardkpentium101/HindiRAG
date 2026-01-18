#!/usr/bin/env python3
"""
Script to configure and test better, more powerful models for the Hindi RAG system
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from better_model_config import update_env_with_better_config, suggest_best_models_for_hindi
from rag_system import HindiRAGSystem
from llm_manager import get_llm_with_provider

def configure_better_models():
    """
    Configure the system to use better, more powerful models
    """
    print("Configuring Hindi RAG System with Better Models")
    print("=" * 60)
    
    # Show model suggestions
    suggestions = suggest_best_models_for_hindi()
    
    print("\nSuggested Models for Hindi RAG System:")
    print("-" * 40)
    
    for category, models in suggestions.items():
        print(f"\n{category.replace('_', ' ').upper()}:")
        for model in models:
            provider, model_name = model.split('.', 1)
            print(f"  • {model} - {get_model_description(provider, model_name)}")
    
    print("\n" + "=" * 60)
    print("Configuration Options:")
    print("-" * 20)
    
    # Configure with a recommended model based on system capabilities
    print("\n1. Configuring with a recommended model...")
    
    # Check if we have commercial API keys available
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_google = bool(os.getenv("GOOGLE_API_KEY"))
    
    if has_openai:
        print("   ✓ OpenAI API key detected - configuring with GPT-4 Turbo")
        update_env_with_better_config(provider="openai", model_key="gpt_4_turbo")
    elif has_anthropic:
        print("   ✓ Anthropic API key detected - configuring with Claude 3 Opus")
        update_env_with_better_config(provider="anthropic", model_key="claude_3_opus")
    elif has_google:
        print("   ✓ Google API key detected - configuring with Gemini 1.5 Pro")
        update_env_with_better_config(provider="google", model_key="gemini_1_5_pro")
    else:
        # For open source models, recommend a good multilingual model
        print("   ℹ No commercial API keys found - configuring with multilingual Mistral model")
        update_env_with_better_config(provider="huggingface", model_key="multilingual_mistral")
    
    print("\n2. Model configuration completed!")
    print("   The .env file has been updated with better model settings.")
    print("   Restart your application to use the new configuration.")


def get_model_description(provider, model_name):
    """
    Get description for a model
    """
    from better_model_config import BETTER_MODEL_CONFIGS
    
    if provider in BETTER_MODEL_CONFIGS and model_name in BETTER_MODEL_CONFIGS[provider]:
        return BETTER_MODEL_CONFIGS[provider][model_name]['description']
    return "Model description not available"


def test_model_performance():
    """
    Test the performance of different models
    """
    print("\n" + "=" * 60)
    print("Testing Model Performance")
    print("=" * 60)
    
    # Test different providers if API keys are available
    providers_to_test = []
    
    if os.getenv("OPENAI_API_KEY"):
        providers_to_test.append(("openai", "gpt_4_turbo"))
    if os.getenv("ANTHROPIC_API_KEY"):
        providers_to_test.append(("anthropic", "claude_3_opus"))
    if os.getenv("GOOGLE_API_KEY"):
        providers_to_test.append(("google", "gemini_1_5_pro"))
    
    # Always test a HuggingFace model as fallback
    providers_to_test.append(("huggingface", "multilingual_mistral"))
    
    test_question = "हिंदी साहित्य में प्रकृति के वर्णन का क्या महत्व है? कुछ उदाहरण सहित समझाइए।"
    
    for provider, model_key in providers_to_test:
        print(f"\nTesting {provider}.{model_key}:")
        print("-" * 30)
        
        try:
            # Get LLM instance
            llm = get_llm_with_provider(provider=provider)
            
            # Simple test - create a basic prompt to test the model
            from langchain_core.prompts import PromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            
            template = f"""Question: {test_question}

Answer the question in Hindi with at least 2-3 sentences."""
            
            prompt = PromptTemplate.from_template(template)
            chain = prompt | llm | StrOutputParser()
            
            response = chain.invoke({})
            
            print(f"Response: {response[:200]}{'...' if len(response) > 200 else ''}")
            print("✓ Model responded successfully")
            
        except Exception as e:
            print(f"✗ Error with {provider}.{model_key}: {str(e)[:100]}")


def show_optimization_tips():
    """
    Show tips for optimizing model performance
    """
    print("\n" + "=" * 60)
    print("Optimization Tips for Better Performance")
    print("=" * 60)
    
    tips = [
        "1. For commercial models (OpenAI, Anthropic, Google):",
        "   • Use the most capable models available (GPT-4 Turbo, Claude 3 Opus, Gemini 1.5 Pro)",
        "   • These provide the best quality responses for complex queries",
        "   • Monitor costs as these models are paid services",
        
        "\n2. For open-source models:",
        "   • Use multilingual models specifically trained for Indian languages",
        "   • ai4bharat/indic-bert for understanding Hindi text",
        "   • mistralai/Mistral-7B-Instruct-v0.2 for generation tasks",
        "   • google/mt5-xxl for complex multilingual tasks",
        
        "\n3. For local models:",
        "   • Use quantized GGUF models for better performance on consumer hardware",
        "   • TheBloke/Mixtral-8x7B-Instruct for powerful local inference",
        "   • Enable GPU acceleration if available (set GPU_LAYERS > 0)",
        
        "\n4. For Ollama:",
        "   • Install and run powerful models locally using 'ollama pull <model>'",
        "   • Recommended: llama3:70b, mixtral:8x7b, mistral-large",
        
        "\n5. System optimizations:",
        "   • Increase MAX_NEW_TOKENS to 1024 for longer responses",
        "   • Adjust TEMPERATURE to 0.3-0.5 for more consistent responses",
        "   • Increase CONTEXT_LENGTH to 8192 for better context handling",
        "   • Use TOP_P=0.9 and TOP_K=50 for better generation quality"
    ]
    
    for tip in tips:
        print(tip)


if __name__ == "__main__":
    print("Hindi RAG System - Better Model Configuration Tool")
    print("=" * 60)
    
    configure_better_models()
    test_model_performance()
    show_optimization_tips()
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. If using commercial models, ensure your API keys are set in .env")
    print("2. For local models, download the model files before running")
    print("3. Restart your application to use the new model configuration")
    print("4. Run 'python main.py' to start the Hindi RAG system with better models")