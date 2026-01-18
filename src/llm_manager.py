import os
from typing import Optional
from dotenv import load_dotenv
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

load_dotenv()

class LLMManager:
    def __init__(self, provider=None, model_kwargs=None):
        self.provider = provider or os.getenv("LLM_PROVIDER", "huggingface")
        self.model_kwargs = model_kwargs or {}
        self.llm_instance = None

    def get_llm(self):
        """
        Get the appropriate LLM instance based on the configured provider
        """
        if self.llm_instance is not None:
            return self.llm_instance

        if self.provider == "openai":
            return self._get_openai_llm()
        elif self.provider == "huggingface":
            return self._get_huggingface_llm()
        elif self.provider == "local":
            return self._get_local_llm()
        elif self.provider == "anthropic":
            return self._get_anthropic_llm()
        elif self.provider == "google":
            return self._get_google_llm()
        elif self.provider == "ollama":
            return self._get_ollama_llm()
        else:
            # Default to HuggingFace
            return self._get_huggingface_llm()

    def _get_openai_llm(self):
        """
        Initialize OpenAI LLM
        """
        from langchain_openai import ChatOpenAI
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        model_name = self.model_kwargs.get("model", os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))

        self.llm_instance = ChatOpenAI(
            model=model_name,
            temperature=float(self.model_kwargs.get("temperature", os.getenv("TEMPERATURE", "0.7"))),
            api_key=openai_api_key
        )
        return self.llm_instance

    def _get_huggingface_llm(self):
        """
        Initialize HuggingFace open-source LLM with support for more powerful models
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        from langchain_community.llms import HuggingFacePipeline

        # Use a more appropriate model for Hindi/multilingual text
        model_name = self.model_kwargs.get("model", os.getenv("HUGGINGFACE_MODEL", "facebook/blenderbot-400M-distill"))

        # Determine if we're using a model that requires special handling
        is_seq_to_seq = any(name in model_name.lower() for name in ['t5', 'mt5', 'bart', 'marian'])
        task = "text2text-generation" if is_seq_to_seq else "text-generation"

        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Check if we should use a specific device
        device_map = self.model_kwargs.get("device_map", "auto" if torch.cuda.is_available() else "cpu")

        # Load model with proper handling for different architectures
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # Use float16 for GPU to save memory
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True if "bloom" not in model_name.lower() else False,  # Some models like BLOOM have issues with low_cpu_mem_usage
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else None  # Use flash attention if available
        )

        # Create text generation pipeline with enhanced parameters for better models
        pipe = pipeline(
            task,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=int(self.model_kwargs.get("max_new_tokens", os.getenv("MAX_NEW_TOKENS", "512"))),  # Increased for better models
            temperature=float(self.model_kwargs.get("temperature", os.getenv("TEMPERATURE", "0.7"))),
            repetition_penalty=float(self.model_kwargs.get("repetition_penalty", os.getenv("REPETITION_PENALTY", "1.1"))),
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_full_text=False,  # Don't return the prompt in the output
            # Additional parameters for better models
            top_p=float(self.model_kwargs.get("top_p", os.getenv("TOP_P", "0.9"))),
            top_k=int(self.model_kwargs.get("top_k", os.getenv("TOP_K", "50"))),
        )

        # Create LangChain compatible LLM
        self.llm_instance = HuggingFacePipeline(pipeline=pipe)
        return self.llm_instance

    def _get_local_llm(self):
        """
        Initialize local LLM with support for more powerful models
        """
        try:
            # Try the newer import path first
            from langchain_community.llms import CTransformers
        except ImportError:
            # Fallback to older import path
            from langchain.llms import CTransformers

        model_path = self.model_kwargs.get("model_path", os.getenv("LOCAL_MODEL_PATH", "TheBloke/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q4_K_M.gguf"))
        model_type = self.model_kwargs.get("model_type", os.getenv("LOCAL_MODEL_TYPE", "mistral"))

        # Enhanced configuration for more powerful local models
        config = {
            'max_new_tokens': int(self.model_kwargs.get("max_new_tokens", os.getenv("MAX_NEW_TOKENS", "512"))),  # Increased for better responses
            'temperature': float(self.model_kwargs.get("temperature", os.getenv("TEMPERATURE", "0.7"))),
            'context_length': int(self.model_kwargs.get("context_length", os.getenv("CONTEXT_LENGTH", "4096"))),  # Increased for better context
            'gpu_layers': int(self.model_kwargs.get("gpu_layers", os.getenv("GPU_LAYERS", "0"))),  # Enable GPU acceleration if available
            'batch_size': int(self.model_kwargs.get("batch_size", os.getenv("BATCH_SIZE", "512"))),
            'seed': int(self.model_kwargs.get("seed", os.getenv("SEED", "-1"))),
            # Additional parameters for better performance
            'top_k': int(self.model_kwargs.get("top_k", os.getenv("TOP_K", "50"))),
            'top_p': float(self.model_kwargs.get("top_p", os.getenv("TOP_P", "0.95"))),
            'repeat_penalty': float(self.model_kwargs.get("repeat_penalty", os.getenv("REPEAT_PENALTY", "1.1"))),
            'n_threads': int(self.model_kwargs.get("n_threads", os.getenv("N_THREADS", "0"))),  # Use all CPU cores if 0
        }

        self.llm_instance = CTransformers(
            model=model_path,
            model_type=model_type,
            config=config
        )
        return self.llm_instance

    def _get_anthropic_llm(self):
        """
        Initialize Anthropic Claude LLM
        """
        from langchain_anthropic import ChatAnthropic
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")

        model_name = self.model_kwargs.get("model", os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"))

        self.llm_instance = ChatAnthropic(
            model=model_name,
            temperature=float(self.model_kwargs.get("temperature", os.getenv("TEMPERATURE", "0.7"))),
            api_key=anthropic_api_key
        )
        return self.llm_instance

    def _get_google_llm(self):
        """
        Initialize Google Gemini LLM
        """
        from langchain_google_genai import ChatGoogleGenerativeAI
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")

        model_name = self.model_kwargs.get("model", os.getenv("GOOGLE_MODEL", "gemini-1d.5-flash"))

        self.llm_instance = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=float(self.model_kwargs.get("temperature", os.getenv("TEMPERATURE", "0.7"))),
            google_api_key=google_api_key
        )
        return self.llm_instance

    def _get_ollama_llm(self):
        """
        Initialize Ollama LLM
        """
        from langchain_ollama import ChatOllama

        model_name = self.model_kwargs.get("model", os.getenv("OLLAMA_MODEL", "llama3"))

        self.llm_instance = ChatOllama(
            model=model_name,
            temperature=float(self.model_kwargs.get("temperature", os.getenv("TEMPERATURE", "0.7"))),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        return self.llm_instance

# Global instance
llm_manager = LLMManager()

def get_llm():
    """
    Convenience function to get the configured LLM
    """
    return llm_manager.get_llm()

def get_llm_with_provider(provider=None, model_kwargs=None):
    """
    Get LLM with specific provider and model kwargs
    """
    manager = LLMManager(provider=provider, model_kwargs=model_kwargs)
    return manager.get_llm()