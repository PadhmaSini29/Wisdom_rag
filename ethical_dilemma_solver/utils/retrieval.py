# retrieval.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL_NAME, TOP_KURALS
from data_loader import load_thirukkural_data

# --------------------------
# Load Thirukkural Dataset
# --------------------------
print("📖 [retrieval.py] Loading Thirukkural dataset...")
kurals = load_thirukkural_data(use_local=True)

# --------------------------
# Load Embedding Model
# --------------------------
print("🔄 [retrieval.py] Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# --------------------------
# Embed All Kurals
# --------------------------
print("🧠 [retrieval.py] Encoding all Kurals...")
kural_embeddings = embedder.encode(kurals, convert_to_tensor=True).cpu().detach().numpy()

# --------------------------
# Retrieval Function (cosine similarity)
# --------------------------
# utils/retrieval.py

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def retrieve_kurals(query: str, kurals, kural_embeddings, embedder, top_k: int = 3) -> list:
    if not query.strip():
        raise ValueError("❌ Query is empty.")

    if not kurals or kural_embeddings.shape[0] == 0:
        raise ValueError("❌ Kural data or embeddings are missing.")

    # ✅ Correct: pass [query] to get shape (1, dim)
    query_embedding = embedder.encode([query])

    # ✅ Now both arrays are 2D
    similarities = cosine_similarity(query_embedding, kural_embeddings)  # shape (1, N)
    top_indices = np.argsort(similarities[0])[-top_k:][::-1]
    return [kurals[i] for i in top_indices]



