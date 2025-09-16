#!/bin/bash

echo "Setting up RAG system with ClickHouse"
echo "====================================="

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Check if ClickHouse is running
echo "Checking ClickHouse connection..."
if command -v clickhouse-client &> /dev/null; then
    echo "ClickHouse client found. Testing connection..."
    clickhouse-client --query "SELECT 1" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✓ ClickHouse is running and accessible"
    else
        echo "⚠️  ClickHouse client found but connection failed"
        echo "Make sure ClickHouse server is running"
    fi
else
    echo "⚠️  ClickHouse client not found"
    echo "Please install ClickHouse or use Docker:"
    echo "docker run -d --name clickhouse-server -p 8123:8123 -p 9000:9000 clickhouse/clickhouse-server"
fi

# Check if Ollama is running
echo "Checking Ollama..."
if command -v ollama &> /dev/null; then
    echo "✓ Ollama found"
    # Check if mistral model is available
    ollama list | grep -q "mistral"
    if [ $? -eq 0 ]; then
        echo "✓ Mistral model found"
    else
        echo "⚠️  Mistral model not found. Run: ollama pull mistral"
    fi
else
    echo "⚠️  Ollama not found. Please install from https://ollama.ai"
fi

echo ""
echo "Setup complete! Run 'python test_rag.py' to test the system."
