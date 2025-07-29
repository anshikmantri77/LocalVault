# LocalVault

A local, personalized chatbot built with Ollama, RAG, and Streamlit.

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Set up Ollama: `bash scripts/setup_ollama.sh`
3. Add your documents to `data/raw/`
4. Run ingestion: `python scripts/ingest_data.py`
5. Fine-tune model: `python scripts/train_model.py`
6. Start server: `python scripts/run_server.py`
7. Launch UI: `streamlit run frontend/streamlit_app.py`

## Features

- Multi-format document processing (PDF, DOCX, CSV, images)
- Vector-based retrieval with ChromaDB
- Fine-tuned LLM with LoRA
- RESTful API with authentication
- Real-time chat interface
