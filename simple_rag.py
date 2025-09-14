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

DOCS_DIR = "docs"         # folder with .txt files
PERSIST_DIR = "chromadb"  # local chroma persistence

def load_documents(folder):
    docs = []
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".txt"):
            continue
        path = os.path.join(folder, fname)
        loader = TextLoader(path, encoding="utf-8")
        docs.extend(loader.load())
    return docs

def build_or_load_store(docs):
    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="mistral")  # you can swap to llama2, etc.
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    vectordb.persist()
    return vectordb.as_retriever(search_kwargs={"k": 4})

def main():
    if not os.path.isdir(DOCS_DIR):
        print(f"Create a '{DOCS_DIR}' folder and add some .txt files to index.")
        return

    print("Loading documents...")
    docs = load_documents(DOCS_DIR)
    if not docs:
        print("No .txt documents found in docs/ — add some and rerun.")
        return

    print("Building vectorstore...")
    retriever = build_or_load_store(docs)

    # Use Ollama as LLM
    llm = Ollama(model="mistral", temperature=0)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    print("Ready. Ask a question (type 'exit' to quit).")
    while True:
        q = input("\nQuestion> ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue
        ans = qa.run(q)
        print("\nAnswer:\n", ans)

if __name__ == "__main__":
    main()
