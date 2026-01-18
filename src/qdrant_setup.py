import qdrant_client
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import os
from dotenv import load_dotenv

load_dotenv()

class QdrantSetup:
    def __init__(self, host=None, port=None, api_key=None, https=True):
        """
        Initialize Qdrant client - supports both local and cloud instances
        """
        # Check if using cloud instance
        cloud_url = os.getenv("QDRANT_URL")  # For cloud instances
        cloud_api_key = os.getenv("QDRANT_API_KEY")  # For cloud instances

        if cloud_url:
            # Use cloud instance
            self.client = qdrant_client.QdrantClient(
                url=cloud_url,
                api_key=cloud_api_key,
                https=https
            )
        else:
            # Use local instance
            host = host or os.getenv("QDRANT_HOST", "localhost")
            port = port or int(os.getenv("QDRANT_PORT", 6333))
            self.client = qdrant_client.QdrantClient(
                host=host,
                port=port
            )

        self.collection_name = "hindi_poems_stories"
        
    def create_collection(self, vector_size=384):
        """
        Create a collection in Qdrant for storing Hindi text embeddings
        """
        # Check if collection already exists
        collections = self.client.get_collections()
        collection_names = [col.name for col in collections.collections]

        if self.collection_name in collection_names:
            print(f"Collection '{self.collection_name}' already exists.")
            return

        # Create collection with specified vector size
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        
        print(f"Collection '{self.collection_name}' created successfully.")
    
    def get_client(self):
        """
        Return the Qdrant client instance
        """
        return self.client
    
    def get_collection_name(self):
        """
        Return the collection name
        """
        return self.collection_name

if __name__ == "__main__":
    # Initialize Qdrant setup
    qdrant_setup = QdrantSetup()
    
    # Create collection
    qdrant_setup.create_collection()
    
    print("Qdrant setup completed!")