"""
Simple single-file RAG example with Ollama and ClickHouse

How it works:
1. Load text files from ./docs
2. Split into chunks
3. Build embeddings and store in ClickHouse
4. Use Ollama model as LLM for answering
5. Ask questions interactively

Requirements:
  pip install langchain clickhouse-connect ollama

Run Ollama separately (e.g. `ollama serve`), and make sure a model is pulled:
  ollama pull mistral
Then run:
  python simple_rag.py
"""

import os
import json
import random
import string
import clickhouse_connect
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.schema import Document
from typing import List, Optional

DOCS_DIR = "Notes"         # folder with .txt files
QUESTIONS_FILE = "generated_questions.json"  # output file for generated questions

# ClickHouse configuration - you can set these as environment variables
CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "localhost")
CLICKHOUSE_PORT = int(os.getenv("CLICKHOUSE_PORT", "8123"))
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "default")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "")
CLICKHOUSE_DATABASE = os.getenv("CLICKHOUSE_DATABASE", "default")
CLICKHOUSE_TABLE = "documents"

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

def generate_object_id():
    """Generate a random 24-character hex string for MongoDB ObjectId"""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=24))

def generate_question_json(question_text, options, correct_answer, category="", image_url=""):
    """Generate a single question in the specified JSON format"""
    return {
        "_id": {"$oid": generate_object_id()},
        "question": question_text,
        "one": options[0] if len(options) > 0 else "",
        "two": options[1] if len(options) > 1 else "",
        "three": options[2] if len(options) > 2 else "",
        "four": options[3] if len(options) > 3 else "",
        "correct": correct_answer,
        "category": category,
        "QuestionPic": image_url,
        "__v": {"$numberInt": "0"}
    }

def generate_questions_from_content(llm, retriever, num_questions=5):
    """Generate multiple-choice questions based on the indexed content"""
    print(f"\nGenerating {num_questions} questions from your content...")
    
    # Get some random chunks to base questions on
    sample_docs = retriever.get_relevant_documents("")
    if not sample_docs:
        print("No documents found to generate questions from.")
        return []
    
    questions = []
    used_content = set()
    
    for i in range(num_questions):
        # Get a random document chunk
        doc = random.choice(sample_docs)
        content = doc.page_content[:500]  # Limit content length
        
        # Skip if we've already used this content
        if content in used_content:
            continue
        used_content.add(content)
        
        # Generate question using LLM
        prompt = f"""
        Based on the following content, create a multiple-choice question with 4 options.
        Content: {content}
        
        Please provide your response in this exact format:
        QUESTION: [Your question here]
        A: [Option 1]
        B: [Option 2] 
        C: [Option 3]
        D: [Option 4]
        CORRECT: [A, B, C, or D]
        CATEGORY: [Category name]
        """
        
        try:
            response = llm(prompt)
            
            # Parse the response
            lines = response.strip().split('\n')
            question_text = ""
            options = []
            correct_answer = ""
            category = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith("QUESTION:"):
                    question_text = line.replace("QUESTION:", "").strip()
                elif line.startswith("A:"):
                    options.append(line.replace("A:", "").strip())
                elif line.startswith("B:"):
                    options.append(line.replace("B:", "").strip())
                elif line.startswith("C:"):
                    options.append(line.replace("C:", "").strip())
                elif line.startswith("D:"):
                    options.append(line.replace("D:", "").strip())
                elif line.startswith("CORRECT:"):
                    correct_letter = line.replace("CORRECT:", "").strip().upper()
                    correct_answer = {"A": "one", "B": "two", "C": "three", "D": "four"}.get(correct_letter, "one")
                elif line.startswith("CATEGORY:"):
                    category = line.replace("CATEGORY:", "").strip()
            
            if question_text and len(options) >= 4 and correct_answer:
                question_json = generate_question_json(question_text, options, correct_answer, category)
                questions.append(question_json)
                print(f"Generated question {i+1}: {question_text[:50]}...")
            else:
                print(f"Failed to parse question {i+1}, skipping...")
                
        except Exception as e:
            print(f"Error generating question {i+1}: {e}")
            continue
    
    return questions

def save_questions_to_json(questions, filename=QUESTIONS_FILE):
    """Save questions to JSON file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(questions)} questions to {filename}")
    except Exception as e:
        print(f"Error saving questions: {e}")

class ClickHouseVectorStore:
    """ClickHouse-based vector store for RAG"""
    
    def __init__(self, host: str, port: int, user: str, password: str, database: str, table: str, embedding_function):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.table = table
        self.embedding_function = embedding_function
        self.client = None
        self._connect()
        self._create_database()
        self._create_table()
    
    def _connect(self):
        """Connect to ClickHouse"""
        try:
            self.client = clickhouse_connect.get_client(
                host=self.host,
                port=self.port,
                username=self.user,
                password=self.password
            )
            print(f"Connected to ClickHouse at {self.host}:{self.port}")
        except Exception as e:
            print(f"Error connecting to ClickHouse: {e}")
            raise
    
    def _create_database(self):
        """Create database if it doesn't exist"""
        try:
            self.client.command(f"CREATE DATABASE IF NOT EXISTS {self.database}")
            self.client.command(f"USE {self.database}")
            print(f"Database {self.database} ready")
        except Exception as e:
            print(f"Error creating database: {e}")
            raise
    
    def _create_table(self):
        """Create table for storing documents and embeddings"""
        try:
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                id String,
                content String,
                metadata String,
                source String,
                embedding Array(Float32),
                created_at DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            ORDER BY id
            """
            self.client.command(create_table_sql)
            print(f"Table {self.table} ready")
        except Exception as e:
            print(f"Error creating table: {e}")
            raise
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store"""
        try:
            print(f"Adding {len(documents)} documents to ClickHouse...")
            
            # Prepare data for batch insert
            data = []
            for doc in documents:
                # Generate embedding
                embedding = self.embedding_function.embed_query(doc.page_content)
                
                # Prepare metadata
                metadata = json.dumps(doc.metadata) if doc.metadata else "{}"
                
                data.append({
                    'id': f"{doc.metadata.get('source', 'unknown')}_{hash(doc.page_content)}",
                    'content': doc.page_content,
                    'metadata': metadata,
                    'source': doc.metadata.get('source', 'unknown'),
                    'embedding': embedding
                })
            
            # Batch insert
            self.client.insert(self.table, data)
            print(f"Successfully added {len(documents)} documents")
            
        except Exception as e:
            print(f"Error adding documents: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents using cosine similarity"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_function.embed_query(query)
            
            # Use ClickHouse's cosineDistance function for similarity search
            search_sql = f"""
            SELECT 
                id,
                content,
                metadata,
                source,
                cosineDistance(embedding, {query_embedding}) as distance
            FROM {self.table}
            ORDER BY distance ASC
            LIMIT {k}
            """
            
            result = self.client.query(search_sql)
            
            documents = []
            for row in result.result_rows:
                doc_id, content, metadata_str, source, distance = row
                metadata = json.loads(metadata_str) if metadata_str else {}
                metadata['distance'] = distance
                
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    def get_relevant_documents(self, query: str, k: int = 4) -> List[Document]:
        """Get relevant documents for a query"""
        return self.similarity_search(query, k)
    
    def as_retriever(self, search_kwargs: dict = None):
        """Return a retriever object"""
        if search_kwargs is None:
            search_kwargs = {"k": 4}
        
        class ClickHouseRetriever:
            def __init__(self, vector_store, search_kwargs):
                self.vector_store = vector_store
                self.search_kwargs = search_kwargs
            
            def get_relevant_documents(self, query: str) -> List[Document]:
                return self.vector_store.get_relevant_documents(query, self.search_kwargs.get("k", 4))
        
        return ClickHouseRetriever(self, search_kwargs)

def build_or_load_store(docs):
    print("Splitting documents into chunks...")
    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    print(f"Created {len(split_docs)} chunks")

    print("Initializing embeddings...")
    try:
        embeddings = OllamaEmbeddings(model="mistral")  # you can swap to llama2, etc.
        print("Building ClickHouse vector store (this may take a while)...")
        
        # Create ClickHouse vector store
        vectordb = ClickHouseVectorStore(
            host=CLICKHOUSE_HOST,
            port=CLICKHOUSE_PORT,
            user=CLICKHOUSE_USER,
            password=CLICKHOUSE_PASSWORD,
            database=CLICKHOUSE_DATABASE,
            table=CLICKHOUSE_TABLE,
            embedding_function=embeddings
        )
        
        # Add documents to ClickHouse
        vectordb.add_documents(split_docs)
        print("ClickHouse vector store ready!")
        return vectordb.as_retriever(search_kwargs={"k": 4})
    except Exception as e:
        print(f"Error building vector store: {e}")
        raise

def get_clickhouse_config():
    """Get ClickHouse configuration from user or environment"""
    global CLICKHOUSE_HOST, CLICKHOUSE_PORT, CLICKHOUSE_USER, CLICKHOUSE_PASSWORD, CLICKHOUSE_DATABASE
    
    print("ClickHouse Configuration")
    print("=" * 30)
    
    # Check if environment variables are set (including empty password)
    if 'CLICKHOUSE_HOST' in os.environ:
        print(f"Using environment configuration:")
        print(f"  Host: {CLICKHOUSE_HOST}")
        print(f"  Port: {CLICKHOUSE_PORT}")
        print(f"  User: {CLICKHOUSE_USER}")
        print(f"  Database: {CLICKHOUSE_DATABASE}")
        return
    
    # Get configuration from user
    print("Please provide ClickHouse connection details:")
    host = input(f"Host [{CLICKHOUSE_HOST}]: ").strip() or CLICKHOUSE_HOST
    port = input(f"Port [{CLICKHOUSE_PORT}]: ").strip() or str(CLICKHOUSE_PORT)
    user = input(f"User [{CLICKHOUSE_USER}]: ").strip() or CLICKHOUSE_USER
    password = input("Password: ").strip()
    database = input(f"Database [{CLICKHOUSE_DATABASE}]: ").strip() or CLICKHOUSE_DATABASE
    
    # Update global variables
    CLICKHOUSE_HOST = host
    CLICKHOUSE_PORT = int(port)
    CLICKHOUSE_USER = user
    CLICKHOUSE_PASSWORD = password
    CLICKHOUSE_DATABASE = database

def main():
    if not os.path.isdir(DOCS_DIR):
        print(f"Create a '{DOCS_DIR}' folder and add some .md files to index.")
        return

    try:
        # Get ClickHouse configuration
        get_clickhouse_config()
        
        print("\nLoading documents...")
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

        print("\n" + "="*50)
        print("RAG System Ready!")
        print("="*50)
        print("Commands:")
        print("  - Ask questions normally")
        print("  - Type 'generate' to create questions from your content")
        print("  - Type 'exit' to quit")
        print("="*50)
        
        while True:
            try:
                q = input("\nQuestion> ").strip()
                if q.lower() in ("exit", "quit"):
                    break
                elif q.lower() == "generate":
                    # Generate questions from content
                    num_questions = input("How many questions to generate? (default: 5): ").strip()
                    try:
                        num_questions = int(num_questions) if num_questions else 5
                    except ValueError:
                        num_questions = 5
                    
                    questions = generate_questions_from_content(llm, retriever, num_questions)
                    if questions:
                        save_questions_to_json(questions)
                    else:
                        print("No questions were generated.")
                    continue
                elif not q:
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
