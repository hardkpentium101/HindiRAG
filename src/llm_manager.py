"""
Updated LLM Manager module with GLM model support for the Hindi RAG system
"""
from typing import Optional, Dict, Any
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModel
import torch
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class LLMManager:
    _instance = None
    _llm_instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMManager, cls).__new__(cls)
        return cls._instance

    def get_llm(self, provider: str = "huggingface", model_kwargs: Optional[Dict[str, Any]] = None):
        """
        Get LLM instance based on provider
        """
        if provider == "huggingface":
            return self._get_huggingface_llm(model_kwargs)
        else:
            # Default to huggingface
            return self._get_huggingface_llm(model_kwargs)

    def _get_huggingface_llm(self, model_kwargs: Optional[Dict[str, Any]] = None):
        """
        Get HuggingFace LLM with proper configuration to handle sequence length issues
        Updated to use GLM model for better Hindi text generation
        """
        if self._llm_instance is not None:
            return self._llm_instance

        # Use a GLM model that's better for Hindi text generation
        # Using THUDM/chatglm3-6b as it's known for good multilingual support
        model_name = os.getenv("HUGGINGFACE_MODEL", "THUDM/chatglm3-6b")

        try:
            # Special handling for ChatGLM models which have their own implementation
            if "chatglm" in model_name.lower():
                print(f"Initializing ChatGLM model: {model_name}")
                
                # For ChatGLM models, we need to use the specific model class
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                
                model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    device_map="cpu",  # Using CPU to avoid GPU memory issues
                    torch_dtype=torch.float32 if torch.cuda.is_available() else torch.float32,
                    low_cpu_mem_usage=False
                )
                
                # Move model to CPU if not already there
                model = model.to('cpu')
                model = model.eval()  # Set to evaluation mode
                
                # Create a custom pipeline for ChatGLM since it has specific requirements
                def chatglm_generate(prompt, **kwargs):
                    response, history = model.chat(
                        tokenizer,
                        prompt,
                        history=[],
                        **kwargs
                    )
                    return [{"generated_text": response}]
                
                # Wrap the model in a way that's compatible with LangChain
                class ChatGLMPipeline:
                    def __init__(self, model, tokenizer):
                        self.model = model
                        self.tokenizer = tokenizer
                    
                    def __call__(self, inputs, **kwargs):
                        # Process the inputs for ChatGLM
                        if isinstance(inputs, str):
                            inputs = [inputs]
                        
                        results = []
                        for inp in inputs:
                            response, _ = self.model.chat(
                                self.tokenizer,
                                inp,
                                history=[],
                                max_length=kwargs.get('max_new_tokens', int(os.getenv("MAX_NEW_TOKENS", 1024))),
                                temperature=kwargs.get('temperature', float(os.getenv("TEMPERATURE", 0.7))),
                                top_p=kwargs.get('top_p', float(os.getenv("TOP_P", 0.9))),
                                top_k=kwargs.get('top_k', int(os.getenv("TOP_K", 50))),
                                repetition_penalty=kwargs.get('repetition_penalty', float(os.getenv("REPETITION_PENALTY", 1.1))),
                            )
                            results.append({"generated_text": response})
                        return results
                
                pipe = ChatGLMPipeline(model, tokenizer)
                from langchain_community.llms import HuggingFacePipeline
                self._llm_instance = HuggingFacePipeline(pipeline=pipe)
                print(f"ChatGLM LLM initialized successfully with {model_name}")
                return self._llm_instance
            else:
                # For non-ChatGLM models, use the standard pipeline
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

                # Add padding token if it doesn't exist
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                # Check if the model is already initialized on meta device
                # and handle it properly to avoid the meta tensor error
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="cpu",  # Using CPU to avoid GPU memory issues
                    trust_remote_code=True,
                    torch_dtype=torch.float32 if torch.cuda.is_available() else torch.float32,  # Specify dtype explicitly
                    low_cpu_mem_usage=False  # Avoid issues with meta tensors
                )

                # Ensure model is properly initialized on the target device
                model = model.to('cpu')

                # Create the text generation pipeline with parameters optimized for Hindi text
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", 1024)),  # Increased length for better responses
                    do_sample=True,
                    temperature=float(os.getenv("TEMPERATURE", 0.7)),  # Balanced creativity for Hindi text
                    top_k=int(os.getenv("TOP_K", 50)),  # Larger sampling pool
                    top_p=float(os.getenv("TOP_P", 0.9)),  # Good balance for diversity
                    repetition_penalty=float(os.getenv("REPETITION_PENALTY", 1.1)),  # Penalty to avoid repetitions
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    truncation=True,  # Enable truncation to handle long inputs
                    return_full_text=False,  # Only return generated text, not the prompt
                    clean_up_tokenization_spaces=True  # Better output formatting
                )

                self._llm_instance = HuggingFacePipeline(pipeline=pipe)
                print(f"LLM initialized successfully with {model_name}")
                return self._llm_instance

        except Exception as e:
            print(f"Failed to initialize LLM with {model_name}: {e}")
            # Try with a simpler model as fallback
            try:
                print("Trying fallback with THUDM/chatglm3-6b...")
                model_name = "THUDM/chatglm3-6b"
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                
                model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=False
                )
                
                model = model.to('cpu')
                model = model.eval()
                
                # Create ChatGLM pipeline wrapper
                class ChatGLMPipeline:
                    def __init__(self, model, tokenizer):
                        self.model = model
                        self.tokenizer = tokenizer
                    
                    def __call__(self, inputs, **kwargs):
                        if isinstance(inputs, str):
                            inputs = [inputs]
                        
                        results = []
                        for inp in inputs:
                            response, _ = self.model.chat(
                                self.tokenizer,
                                inp,
                                history=[],
                                max_length=kwargs.get('max_new_tokens', int(os.getenv("MAX_NEW_TOKENS", 1024))),
                                temperature=kwargs.get('temperature', float(os.getenv("TEMPERATURE", 0.7))),
                                top_p=kwargs.get('top_p', float(os.getenv("TOP_P", 0.9))),
                                top_k=kwargs.get('top_k', int(os.getenv("TOP_K", 50))),
                                repetition_penalty=kwargs.get('repetition_penalty', float(os.getenv("REPETITION_PENALTY", 1.1))),
                            )
                            results.append({"generated_text": response})
                        return results
                
                pipe = ChatGLMPipeline(model, tokenizer)
                self._llm_instance = HuggingFacePipeline(pipeline=pipe)
                print("Fallback ChatGLM LLM initialized successfully")
                return self._llm_instance
            except Exception as fallback_e:
                print(f"ChatGLM fallback also failed: {fallback_e}")
                # Final fallback to a standard model
                try:
                    print("Trying final fallback with distilgpt2...")
                    model_name = "distilgpt2"
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token

                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="cpu",
                        trust_remote_code=True,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=False
                    )
                    model = model.to('cpu')

                    pipe = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", 1024)),
                        do_sample=True,
                        temperature=float(os.getenv("TEMPERATURE", 0.7)),
                        top_k=int(os.getenv("TOP_K", 50)),
                        top_p=float(os.getenv("TOP_P", 0.9)),
                        repetition_penalty=float(os.getenv("REPETITION_PENALTY", 1.1)),
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        truncation=True,
                        return_full_text=False
                    )

                    self._llm_instance = HuggingFacePipeline(pipeline=pipe)
                    print("Final fallback LLM initialized successfully with distilgpt2")
                    return self._llm_instance
                except Exception as final_fallback_e:
                    print(f"Final fallback also failed: {final_fallback_e}")
                    return self._get_fallback_llm()

    def _get_fallback_llm(self):
        """
        Fallback LLM implementation that doesn't rely on transformers
        """
        print("Using fallback LLM implementation")
        # Since we can't initialize a proper LLM, we'll create a mock that at least allows the system to function
        # This will be handled in the rag_system by using the fallback methods
        class MockLLM:
            def __init__(self):
                self.model = "fallback"

            def invoke(self, inputs):
                # Return a simple response indicating the fallback
                return "Mock LLM response - actual generation failed"

        return MockLLM()

def get_llm(provider: str = "huggingface", model_kwargs: Optional[Dict[str, Any]] = None):
    """
    Convenience function to get LLM instance
    """
    manager = LLMManager()
    return manager.get_llm(provider, model_kwargs)

def get_llm_with_provider(provider: str = "huggingface", model_kwargs: Optional[Dict[str, Any]] = None):
    """
    Get LLM with specific provider and model kwargs
    """
    manager = LLMManager()
    return manager.get_llm(provider, model_kwargs)