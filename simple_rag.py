"""
Simple single-file RAG example with Ollama

How it works:
1. Load text files from ./docs
2. Split into chunks
3. Build embeddings and store in Chroma
4. Use Ollama model as LLM for answering
5. Ask questions interactively

Requirements:
  pip install langchain chromadb ollama

Run Ollama separately (e.g. `ollama serve`), and make sure a model is pulled:
  ollama pull mistral
Then run:
  python simple_rag.py
"""

import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

DOCS_DIR = "Notes"         # folder with .txt files
PERSIST_DIR = "chromadb"  # local chroma persistence

def load_documents(folder):
    docs = []
    md_files = []
    
    # First, collect all .md files
    for root, dirs, files in os.walk(folder):
        for fname in files:
            if fname.lower().endswith(".md"):
                md_files.append(os.path.join(root, fname))
    
    print(f"Found {len(md_files)} markdown files")
    
    # Process files with progress indicator
    for i, path in enumerate(md_files, 1):
        try:
            print(f"Loading {i}/{len(md_files)}: {os.path.basename(path)}")
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue
    
    print(f"Successfully loaded {len(docs)} documents")
    return docs

def build_or_load_store(docs):
    # Check if vector store already exists
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        print("Loading existing vector store...")
        try:
            embeddings = OllamaEmbeddings(model="mistral")
            vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
            print("Vector store loaded successfully!")
            return vectordb.as_retriever(search_kwargs={"k": 4})
        except Exception as e:
            print(f"Error loading existing vector store: {e}")
            print("Will rebuild vector store...")
    
    print("Splitting documents into chunks...")
    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    print(f"Created {len(split_docs)} chunks")

    print("Initializing embeddings...")
    try:
        embeddings = OllamaEmbeddings(model="mistral")  # you can swap to llama2, etc.
        print("Building vector store (this may take a while)...")
        vectordb = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )
        print("Persisting vector store...")
        vectordb.persist()
        print("Vector store ready!")
        return vectordb.as_retriever(search_kwargs={"k": 4})
    except Exception as e:
        print(f"Error building vector store: {e}")
        raise

def main():
    if not os.path.isdir(DOCS_DIR):
        print(f"Create a '{DOCS_DIR}' folder and add some .md files to index.")
        return

    try:
        print("Loading documents...")
        docs = load_documents(DOCS_DIR)
        if not docs:
            print("No .md documents found in Notes/ — add some and rerun.")
            return

        print("Building vectorstore...")
        retriever = build_or_load_store(docs)

        print("Initializing LLM...")
        # Use Ollama as LLM
        llm = Ollama(model="mistral", temperature=0)

        print("Setting up QA chain...")
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

        print("Ready. Ask a question (type 'exit' to quit).")
        while True:
            try:
                q = input("\nQuestion> ").strip()
                if q.lower() in ("exit", "quit"):
                    break
                if not q:
                    continue
                print("Thinking...")
                ans = qa.run(q)
                print("\nAnswer:\n", ans)
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error processing question: {e}")
                continue
    except Exception as e:
        print(f"Fatal error: {e}")
        return

if __name__ == "__main__":
    main()
