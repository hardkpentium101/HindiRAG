import time
from qdrant_client import QdrantClient

def test_qdrant_connection():
    try:
        print("Attempting to connect to Qdrant...")
        client = QdrantClient(host="localhost", port=6333)
        
        # Wait a bit for connection to establish
        time.sleep(2)
        
        # Try to get collections to test the connection
        collections = client.get_collections()
        print(f"Connected successfully! Found {len(collections.collections)} collections")
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

if __name__ == "__main__":
    test_qdrant_connection()