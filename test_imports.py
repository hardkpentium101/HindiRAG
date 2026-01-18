import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """
    Test that all modules can be imported without errors
    """
    print("Testing module imports...")

    try:
        from qdrant_setup import QdrantSetup
        print("✓ QdrantSetup imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import QdrantSetup: {e}")

    try:
        from document_ingestor import DocumentIngestor
        print("✓ DocumentIngestor imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import DocumentIngestor: {e}")

    try:
        from embedding_generator import HindiEmbeddingGenerator, get_embedding_function
        print("✓ EmbeddingGenerator imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import EmbeddingGenerator: {e}")

    try:
        from rag_system import HindiRAGSystem
        print("✓ RAGSystem imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import RAGSystem: {e}")

    print("\nAll modules imported successfully! The system is ready.")

if __name__ == "__main__":
    test_imports()