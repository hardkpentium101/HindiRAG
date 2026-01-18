# Hindi RAG System

A Retrieval-Augmented Generation (RAG) system for Hindi poems and stories using Qdrant as the vector database, Streamlit for the frontend, and OpenAI as the LLM.

## Features

- Search and query Hindi literature (poems and stories)
- Multilingual embedding support for Hindi text
- Interactive Streamlit frontend
- Qdrant vector database for efficient similarity search
- Support for multiple powerful LLM providers:
  - OpenAI (GPT-4, GPT-4 Turbo)
  - Anthropic (Claude 3 Opus, Sonnet, Haiku)
  - Google (Gemini 1.5 Pro, Gemini Pro)
  - Hugging Face (Mistral, Llama 3, MT5-XXL, IndicBERT)
  - Local models (GGUF format)
  - Ollama (locally hosted models)

## Prerequisites

- Python 3.8+
- Docker (for running Qdrant locally) - optional but recommended

## Setup Instructions

### 1. Clone and Navigate to Project

```bash
cd hindi_rag
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Copy the example environment file and add your OpenAI API key:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```bash
OPENAI_API_KEY=your_actual_openai_api_key_here
```

### 4. Set Up Qdrant

You have two options:

#### Option A: Run Qdrant Locally with Docker (Recommended)

```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

#### Option B: Use Qdrant Cloud

Update your `.env` file with your Qdrant cloud credentials:

```bash
QDRANT_HOST=your-qdrant-host-url
QDRANT_PORT=your-port
QDRANT_API_KEY=your-api-key
```

### 5. Prepare Your Data

Place your Hindi poems and stories in the `data/` directory in either:

- **JSON format**: Each file can contain one or multiple documents with the following structure:
  ```json
  {
    "title": "Title of the poem/story",
    "author": "Author name",
    "text": "Full text of the poem/story",
    "genre": "poem" or "story"
  }
  ```
  
- **TXT format**: Plain text files with poems/stories separated by double newlines

### 6. Run the Application

#### Option A: Run the Full System

```bash
python main.py
```

This will:
- Set up Qdrant collection
- Load and ingest documents from the data directory
- Initialize the RAG system
- Run a sample query

#### Option B: Run the Streamlit Frontend

```bash
streamlit run frontend/app.py
```

Then navigate to `http://localhost:8501` in your browser.

## Architecture

- `src/qdrant_setup.py`: Sets up the Qdrant client and collection
- `src/document_ingestor.py`: Loads and ingests documents into Qdrant
- `src/embedding_generator.py`: Generates embeddings for Hindi text
- `src/rag_system.py`: Implements the RAG retrieval and generation logic
- `frontend/app.py`: Streamlit frontend interface
- `main.py`: Main script to tie everything together

## Usage

1. After starting the Streamlit app, configure Qdrant settings in the sidebar
2. Load your documents by specifying the data directory path
3. Initialize the RAG system
4. Ask questions about your Hindi literature in the main panel

## Customization

- Modify the embedding model in `src/embedding_generator.py` if needed
- Adjust the number of retrieved documents (`top_k`) in the frontend
- Add more sophisticated chunking strategies in `src/document_ingestor.py`

## Using Better, More Powerful Models

The system supports multiple LLM providers with various models of different capabilities. For better responses, consider using:

### Commercial Models (Best Quality)
- **OpenAI**: Use `gpt-4-turbo` or `gpt-4` for the most capable responses
- **Anthropic**: Use `claude-3-opus-20240229` for excellent reasoning capabilities
- **Google**: Use `gemini-1.5-pro-latest` for advanced multimodal understanding

### Open Source Models (Good Balance)
- **Mistral**: `mistralai/Mistral-7B-Instruct-v0.2` for strong multilingual capabilities
- **Llama 3**: `unsloth/Llama-3.2-3B-Instruct` for efficient performance
- **MT5**: `google/mt5-xxl` for powerful text-to-text generation

### Local Models (Privacy & Control)
- **Llama 3 70B**: `TheBloke/Llama-3-70B-Instruct-GGUF` for maximum local capability
- **Mixtral 8x7B**: `TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF` for MoE efficiency
- **Ollama**: Run `ollama pull llama3:70b` or `ollama pull mixtral:8x7b` for local hosting

### Configuration
Update your `.env` file with the desired model:

```bash
# For OpenAI GPT-4 Turbo
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4-turbo-preview

# For Anthropic Claude 3 Opus
LLM_PROVIDER=anthropic
ANTHROPIC_MODEL=claude-3-opus-20240229

# For Hugging Face Mistral
LLM_PROVIDER=huggingface
HUGGINGFACE_MODEL=mistralai/Mistral-7B-Instruct-v0.2

# For local models
LLM_PROVIDER=local
LOCAL_MODEL_PATH=TheBloke/Llama-3-70B-Instruct-GGUF/llama3-70b-instruct.Q4_K_M.gguf
LOCAL_MODEL_TYPE=llama

# For Ollama
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3:70b
```

Run the setup script to configure better models:
```bash
python setup_better_models.py
```

## Troubleshooting

- If you get embedding errors, ensure your OpenAI API key is valid
- If Qdrant connection fails, verify that the Qdrant service is running
- For Hindi text processing issues, check that the multilingual model is properly loaded

## License

MIT