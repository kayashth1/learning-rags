# Introduction to Retrieval-Augmented Generation (RAG)

## What is RAG?
Retrieval-Augmented Generation (RAG) is an AI pattern that combines two steps:

1. **Retrieve**: Fetch relevant information from an external knowledge source (databases, documents, APIs, vector stores, etc.).
2. **Generate**: Feed both the user’s query and the retrieved information into a large language model (LLM) so it can generate a grounded, accurate answer.

This approach ensures the LLM is not limited to its pre-trained knowledge and can use fresh, domain-specific context at query time.

---

## Why RAG?
- **Keeps answers up to date**: Plug in new data without retraining the LLM.  
- **Domain adaptation**: Add your company documents, medical papers, or financial reports.  
- **Reduces hallucinations**: The LLM is guided by retrieved facts.  
- **Cheaper and faster**: No need to fine-tune; just manage embeddings and retrieval.  

---

## How does RAG work?

1. **Document Ingestion**  
   - Collect data (PDFs, text files, CSVs, database rows).  
   - Split into chunks (for example, 500–1000 tokens).  

2. **Embedding and Indexing**  
   - Convert each chunk into a vector using an embedding model.  
   - Store vectors in a vector database (such as Chroma, FAISS, Pinecone, or Weaviate).  

3. **Query and Retrieval**  
   - Convert the user’s query into a vector.  
   - Find the most similar chunks from the database.  

4. **Generation with Augmentation**  
   - Pass both the query and the retrieved chunks to the LLM.  
   - The LLM writes an answer using the retrieved evidence.  

---

## Simple RAG Flow

User Question → Embed → Retrieve (Vector DB) → Context + Question → LLM → Answer

