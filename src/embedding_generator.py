import os
from typing import List
import numpy as np
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
from dotenv import load_dotenv
import torch

load_dotenv()

class HindiEmbeddingGenerator:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize embedding generator for Hindi text
        Using a multilingual model that supports Hindi
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model with proper device handling to avoid meta tensor issues
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32  # Use consistent dtype
        )

        # Initialize OpenAI client for generation
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for Hindi text using multilingual model
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use mean pooling to get the sentence embedding
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        # Convert to list and return
        return embeddings.tolist()
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts
        """
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        """
        Generate response using OpenAI with provided context
        """
        full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nPlease provide a helpful response in Hindi if possible, or in English."
        
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for Hindi literature. Respond appropriately based on the context provided."},
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content

# Alternative implementation using OpenAI embeddings directly
class OpenAIEmbeddingGenerator:
    def __init__(self):
        """
        Initialize OpenAI embedding generator
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "text-embedding-ada-002"
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding from OpenAI
        """
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a batch of texts
        """
        # OpenAI API limits batch size, so we'll process in chunks
        embeddings = []
        chunk_size = 20  # Conservative batch size
        
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            response = self.client.embeddings.create(
                input=chunk,
                model=self.model
            )
            chunk_embeddings = [item.embedding for item in response.data]
            embeddings.extend(chunk_embeddings)
        
        return embeddings

# Choose which embedding generator to use
# For Hindi text, the multilingual transformer model is recommended
def get_embedding_function():
    """
    Return the appropriate embedding function
    """
    # Using the multilingual model which works better for Hindi
    embedder = HindiEmbeddingGenerator()
    return embedder.get_embedding

if __name__ == "__main__":
    # Example usage
    embed_gen = HindiEmbeddingGenerator()
    
    # Test with Hindi text
    hindi_text = "हिंदी साहित्य भारत के समृद्ध साहित्यिक परंपरा का प्रतिनिधित्व करता है।"
    embedding = embed_gen.get_embedding(hindi_text)
    print(f"Embedding length: {len(embedding)}")
    print(f"First 10 values: {embedding[:10]}")