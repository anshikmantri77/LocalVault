# 🔐 LocalVault
### **The Sovereign AI Document Assistant**
**100% Air-Gapped · Production-Grade RAG · Privacy-First Architecture**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/Ollama-Local-orange.svg)](https://ollama.com/)

---

**LocalVault** is a specialized, production-ready Retrieval-Augmented Generation (RAG) assistant designed for users who cannot compromise on data privacy. It transforms your local machine into a high-performance intelligence hub, allowing you to converse with sensitive documents (PDFs, Financial Reports, Resumes, Technical Docs) without a single byte ever leaving your local network.

## ✨ Premium Features

- **🛡️ 100% Sovereign Inference**: Powered by **Ollama** and **Qwen 2.5**. Every "thought" the AI has happens on your silicon, not in the cloud.
- **📊 Surgical Document Parsing**: 
  - **PDF Layout Awareness**: Preserves structural integrity of complex resumes and reports.
  - **Atomic Financial Tagging**: Automatically detects and transposes wide financial tables into searchable atomic units.
  - **OCR Intelligence**: Integrated Tesseract support for scanned images and non-searchable PDFs.
- **⚖️ Dynamic Truth Engine**: Features a multi-stage validation pipeline that cross-references AI synthesis with raw document evidence to eliminate hallucinations.
- **🎯 Hybrid Retrieval (MMR)**: Uses Maximum Marginal Relevance to ensure context diversity, preventing the AI from getting stuck in repetitive document loops.
- **⚡ Real-Time Dashboard**: A sleek, dark-mode Streamlit interface for seamless document management and low-latency interaction.

---

## 🏗️ Technical Architecture

LocalVault is built on a modular **MCP (Model Control Protocol)** inspired architecture:

*   **Brain**: `Qwen 2.5 0.5B` via Ollama (Optimized for speed and local CPU inference).
*   **Memory**: `ChromaDB` (High-performance vector storage).
*   **Senses**: `sentence-transformers/all-mpnet-base-v2` (State-of-the-art semantic embeddings).
*   **Structure**: `FastAPI` Backend + `Streamlit` Frontend.

---

## 🚀 Getting Started

### 1. Prerequisites
- **Ollama**: [Download & Install](https://ollama.com/)
- **Python**: 3.10 or higher.
- **Tesseract OCR** (Optional, for image support): `brew install tesseract` (Mac) or `apt install tesseract-ocr` (Linux).

### 2. Quick Install
```bash
# Clone the repository
git clone https://github.com/anshikmantri77/LocalVault.git
cd LocalVault

# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pull the local brain
ollama pull qwen2.5:0.5b
```

### 3. Launch the Vault
Run the following commands in separate terminals (or use a task runner):

**Terminal 1: The Engine (Backend)**
```bash
python scripts/run_server.py
```

**Terminal 2: The Interface (Frontend)**
```bash
streamlit run frontend/streamlit_app.py --server.port 9000
```

---

## 📁 Supported Ecosystem
- **Documents**: `.pdf`, `.docx`, `.doc`, `.txt`, `.md`
- **Data**: `.csv`, `.xlsx`, `.xls`
- **Images**: `.jpg`, `.jpeg`, `.png` (via OCR)

---

## 🛡️ Security Posture
- **Zero External Calls**: All telemetry, embeddings, and inference are local.
- **Disk Security**: Data is stored in `data/chroma_db` and `data/raw`. We recommend running LocalVault on an encrypted partition.
- **Verification**: The validation pipeline flags any entity mentioned by the AI that cannot be found in your source documents.

---

## 📄 License
Distributed under the **MIT License**. Created with ❤️ by **Anshik Mantri**.
