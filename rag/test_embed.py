from sentence_transformers import SentenceTransformer

# Load embeddings model
model_name = "sentence-transformers/all-MiniLM-L6-v2" # "nvidia/nv-embed-v2"
model = SentenceTransformer(model_name, trust_remote_code=True)

# Function to generate embeddings
def generate_embeddings(texts):
    embeddings = model.encode(texts, convert_to_tensor=False)  # Outputs NumPy arrays
    return embeddings

# Example usage
texts = ["This is a test sentence.", "Here is another example."]
embeddings = generate_embeddings(texts)

# Print embeddings
for i, embedding in enumerate(embeddings):
    print(f"Text: {texts[i]}")
    print(f"Embedding: {embedding}")
    print(f"Embedding Shape: {embedding.shape}")
