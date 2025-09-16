#!/usr/bin/env python3
"""
Test script to debug RAG issues
"""
import os
import sys

def test_imports():
    """Test if all required packages are available"""
    print("Testing imports...")
    try:
        from langchain.text_splitter import CharacterTextSplitter
        print("✓ CharacterTextSplitter imported")
    except ImportError as e:
        print(f"✗ CharacterTextSplitter import failed: {e}")
        return False
    
    try:
        from langchain.document_loaders import TextLoader
        print("✓ TextLoader imported")
    except ImportError as e:
        print(f"✗ TextLoader import failed: {e}")
        return False
    
    try:
        from langchain.vectorstores import Chroma
        print("✓ Chroma imported")
    except ImportError as e:
        print(f"✗ Chroma import failed: {e}")
        return False
    
    try:
        from langchain.chains import RetrievalQA
        print("✓ RetrievalQA imported")
    except ImportError as e:
        print(f"✗ RetrievalQA import failed: {e}")
        return False
    
    try:
        from langchain_community.embeddings import OllamaEmbeddings
        print("✓ OllamaEmbeddings imported")
    except ImportError as e:
        print(f"✗ OllamaEmbeddings import failed: {e}")
        return False
    
    try:
        from langchain_community.llms import Ollama
        print("✓ Ollama imported")
    except ImportError as e:
        print(f"✗ Ollama import failed: {e}")
        return False
    
    return True

def test_ollama_connection():
    """Test if Ollama is accessible"""
    print("\nTesting Ollama connection...")
    try:
        from langchain_community.llms import Ollama
        llm = Ollama(model="mistral", temperature=0)
        # Try a simple test
        response = llm("Hello")
        print(f"✓ Ollama connection successful: {response[:50]}...")
        return True
    except Exception as e:
        print(f"✗ Ollama connection failed: {e}")
        return False

def test_embeddings():
    """Test if embeddings work"""
    print("\nTesting embeddings...")
    try:
        from langchain_community.embeddings import OllamaEmbeddings
        embeddings = OllamaEmbeddings(model="mistral")
        test_text = "This is a test"
        result = embeddings.embed_query(test_text)
        print(f"✓ Embeddings working: {len(result)} dimensions")
        return True
    except Exception as e:
        print(f"✗ Embeddings failed: {e}")
        return False

def test_file_loading():
    """Test loading a single file"""
    print("\nTesting file loading...")
    try:
        from langchain.document_loaders import TextLoader
        # Find first .md file
        for root, dirs, files in os.walk("Notes"):
            for fname in files:
                if fname.endswith(".md"):
                    path = os.path.join(root, fname)
                    print(f"Testing with file: {path}")
                    loader = TextLoader(path, encoding="utf-8")
                    docs = loader.load()
                    print(f"✓ File loaded: {len(docs)} documents, {len(docs[0].page_content)} characters")
                    return True
        print("✗ No .md files found")
        return False
    except Exception as e:
        print(f"✗ File loading failed: {e}")
        return False

def main():
    print("RAG Debug Test")
    print("=" * 50)
    
    if not test_imports():
        print("\n❌ Import test failed. Install required packages:")
        print("pip install langchain chromadb ollama")
        return
    
    if not test_ollama_connection():
        print("\n❌ Ollama test failed. Make sure Ollama is running:")
        print("ollama serve")
        print("ollama pull mistral")
        return
    
    if not test_embeddings():
        print("\n❌ Embeddings test failed.")
        return
    
    if not test_file_loading():
        print("\n❌ File loading test failed.")
        return
    
    print("\n✅ All tests passed! The RAG system should work.")

if __name__ == "__main__":
    main()
