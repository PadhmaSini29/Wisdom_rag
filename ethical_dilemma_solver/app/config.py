# config.py

# -----------------------------
# MODEL CONFIGURATIONS
# -----------------------------

# Embedding model for retrieval
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Generation model (GPT-2 or any compatible causal LM)
GENERATION_MODEL_NAME = "gpt2"

# Maximum number of Kurals to retrieve for context
TOP_KURALS = 3

# Maximum token length for GPT-2 generation
MAX_GENERATION_LENGTH = 200

# Number of responses to generate from GPT-2 (keep 1 for speed)
NUM_RETURN_SEQUENCES = 1


# -----------------------------
# FILE PATHS (for precomputed assets)
# -----------------------------

# FAISS index file path (optional: if saving/loading)
FAISS_INDEX_PATH = "models/faiss_index.bin"

# Embeddings (optional: for caching purposes)
EMBEDDINGS_CACHE_PATH = "models/kural_embeddings.npy"

# Dataset path (if using local version)
THIRUKKURAL_LOCAL_PATH = "data/thirukkural_dataset.json"

# Saved GPT-2 model (optional)
GENERATION_MODEL_PATH = "models/gpt2_model/"  # Optional: for fine-tuned version


# -----------------------------
# UI CONFIGURATIONS
# -----------------------------

# Streamlit App Title
APP_TITLE = "🧠 AI-Based Ethical Dilemma Solver"

# App Description
APP_DESCRIPTION = (
    "Ask a moral or ethical question, and get guidance based on ancient Tamil wisdom "
    "from the Thirukkural using AI (RAG: Retrieval-Augmented Generation)."
)

# Default prompt message
DEFAULT_PROMPT = "Should I prioritize my career over my family obligations?"
