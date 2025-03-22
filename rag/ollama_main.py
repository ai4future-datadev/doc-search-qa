import os
import pickle
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.llms import Ollama

# Load the llm model
model_llm_name = "llama3"
model_llm = Ollama(model=model_llm_name)

# Load embeddings model
model_emb_name = "sentence-transformers/all-MiniLM-L6-v2"
model_emb = SentenceTransformer(model_emb_name, trust_remote_code=True)

# Function to generate embeddings
def generate_embeddings(texts):
    embeddings = model_emb.encode(texts, convert_to_tensor=False)  # Outputs NumPy arrays
    return embeddings

# Step 1: Extract text from PDFs
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()  # Extract text from each page
    return text

# Step 2: Split text into chunks
def split_text_into_chunks(text, chunk_size=500, overlap=100):
    chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# Main pipeline
def index_local_pdfs(folder_path, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    all_chunks = []
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"Processing {pdf_path}")
        text = extract_text_from_pdf(pdf_path)
        chunks = split_text_into_chunks(text)
        all_chunks.extend(chunks)
    # Generate embeddings
    embeddings = generate_embeddings(all_chunks)
    return embeddings, all_chunks

# Query function using cosine similarity
def query_with_cosine_similarity(embeddings, query, chunks, top_k=5):
    query_embedding = generate_embeddings([query])[0]
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    results = [(chunks[i], similarities[i]) for i in top_indices]
    return results # Query Llama 3 through Ollama API


def query_llama(prompt):

    # Query the Llama model via the Ollama API.
    try:
        response = model_llm.generate(prompts=[prompt])
        return response if response else "No response from Llama."
    except Exception as e:
        return f"Error: {str(e)}"


# Configuration
folder_path = r'C:\Users\phili\Meine Ablage\Professor\Bewerbung_Informatik_DHBW_Lörrach\Literatur'

# Indexing PDFs

# Function to save embeddings and chunks to disk
def save_to_disk(embeddings, chunks, embeddings_path, chunks_path):
    with open(embeddings_path, 'wb') as f:
        pickle.dump(embeddings, f)
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)

# Function to load embeddings and chunks from disk
def load_from_disk(embeddings_path, chunks_path):
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
    return embeddings, chunks

# Paths to save/load embeddings and chunks
embeddings_path = 'embeddings.pkl'
chunks_path = 'chunks.pkl'

# Check if embeddings and chunks are already saved to disk
if os.path.exists(embeddings_path) and os.path.exists(chunks_path):
    print("Loading embeddings and chunks from disk...")
    embeddings, chunks = load_from_disk(embeddings_path, chunks_path)
else:
    print("Indexing PDFs...")
    embeddings, chunks = index_local_pdfs(folder_path)
    save_to_disk(embeddings, chunks, embeddings_path, chunks_path)

# Query example
query = "How can you detect IoT routing attacks?"
print("Processing query...")
response = query_with_cosine_similarity(embeddings, query, chunks)

# Prepare the context for Llama
context = " ".join([chunk for chunk, _ in response])
llama_prompt = f"Answer the question based on the following context:\n\nContext:\n{context}\n\nQuestion:\n{query}"

# Get the answer from Llama
print("Querying Llama 3 for an answer...")
llama_answer = query_llama(llama_prompt)
print("Generated Response:")
#print(llama_answer)
print(llama_answer.generations[0][0].text)
