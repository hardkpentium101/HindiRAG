import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from qdrant_setup import QdrantSetup

def recreate_collection():
    """Delete and recreate the collection with correct vector size"""
    print("Recreating collection with correct vector size...")
    
    qdrant_setup = QdrantSetup()
    client = qdrant_setup.get_client()
    collection_name = qdrant_setup.get_collection_name()
    
    # Delete the existing collection
    try:
        client.delete_collection(collection_name=collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except Exception as e:
        print(f"Collection may not have existed or error deleting: {e}")
    
    # Recreate with correct vector size
    qdrant_setup.create_collection(vector_size=384)
    print(f"Created new collection: {collection_name} with vector size 384")

if __name__ == "__main__":
    recreate_collection()