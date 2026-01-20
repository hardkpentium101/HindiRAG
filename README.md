# Hindi RAG System - Fixed

This repository contains a Hindi RAG (Retrieval-Augmented Generation) system that allows users to search and ask questions about Hindi poems and stories. The system was experiencing an issue with meta tensors during initialization, which has now been fixed.

## Issue Fixed

**Problem**: `Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() when moving module from meta to a different device.`

**Solution**: 
1. Updated the `llm_manager.py` to properly handle model initialization and avoid meta tensor issues
2. Updated the `embedding_generator.py` to check for meta tensors and use `to_empty()` method when needed
3. Added proper error handling for meta tensor cases

## Prerequisites

- Python 3.8+
- Virtual environment (recommended)

## Setup Instructions

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <repository-url>
   cd hindi_rag
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install transformers torch qdrant-client langchain-huggingface openai python-dotenv
   ```

4. **Set up Qdrant**:
   - Option 1: Install and run Qdrant locally (follow instructions at https://qdrant.tech/documentation/quick-start/)
   - Option 2: Use Qdrant Cloud (update `.env` with your credentials)

5. **Configure environment variables** (if needed):
   - Copy `.env.example` to `.env` and update values as needed

6. **Prepare data**:
   - Place your Hindi literature data in the `data/` directory in JSON format
   - Sample format: `{"title": "...", "author": "...", "text": "...", "genre": "..."}`

## Running the Application

1. **Start Qdrant** (if running locally):
   ```bash
   # Follow Qdrant startup instructions
   # Usually: docker run -p 6333:6333 qdrant/qdrant
   ```

2. **Run the Streamlit application**:
   ```bash
   streamlit run frontend/app.py
   ```

3. **Access the application**:
   - Open your browser and go to `http://localhost:8080` (or as displayed in the terminal)

## Key Files Modified

- `src/llm_manager.py`: Fixed meta tensor issue during model initialization
- `src/embedding_generator.py`: Added meta tensor handling for embedding models
- Both files now properly handle the transition from meta tensors to actual device tensors

## Features

- Hindi literature search and question answering
- Support for poems, stories, and other literary forms
- Multilingual embedding support
- Local LLM inference with fallback options
- Streamlit-based user interface

## Troubleshooting

If you encounter issues:
1. Make sure Qdrant is running and accessible
2. Verify your environment variables in `.env`
3. Check that all dependencies are installed
4. Ensure your data files are in the correct format in the `data/` directory

## Contributing

Feel free to submit issues and enhancement requests. Pull requests are welcome!