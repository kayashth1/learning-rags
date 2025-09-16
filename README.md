# My Personal RAG Learning Repo

This repository contains my notes, experiments, and code while learning **Retrieval-Augmented Generation (RAG)**.  

## Contents
- Roadmap for learning RAG  
- Simple RAG pipeline examples (Ollama + ClickHouse)  
- Practice projects and experiments  

## Goals
- Understand embeddings, vector databases, and retrievers  
- Build small RAG prototypes with local and hosted LLMs  
- Document my learning journey

## Quick Start

### Prerequisites
1. **Ollama** - Install and run locally:
   ```bash
   # Install Ollama (see https://ollama.ai)
   ollama serve
   ollama pull mistral
   ```

2. **ClickHouse** - Install and run:
   ```bash
   # Using Docker (recommended)
   docker run -d --name clickhouse-server -p 8123:8123 -p 9000:9000 clickhouse/clickhouse-server
   
   # Or install locally (see https://clickhouse.com/docs/en/install)
   ```

3. **Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the RAG System

1. **Test the setup**:
   ```bash
   python test_rag.py
   ```

2. **Run the RAG system**:
   ```bash
   python simple_rag.py
   ```

### Configuration

The system uses environment variables for ClickHouse configuration:

```bash
export CLICKHOUSE_HOST="localhost"
export CLICKHOUSE_PORT="8123"
export CLICKHOUSE_USER="default"
export CLICKHOUSE_PASSWORD="your_password"
export CLICKHOUSE_DATABASE="rag_db"
```

If not set, the system will prompt for these values interactively.

### Features

- **Document Processing**: Automatically loads and chunks markdown files from the `Notes/` directory
- **Vector Storage**: Uses ClickHouse for efficient vector similarity search
- **Question Generation**: Can generate multiple-choice questions from your content
- **Interactive Q&A**: Ask questions about your documents  
