import re
import numpy as np
import os
from datasets import load_dataset
from config import (
    EMBEDDING_MODEL_NAME,
    TOP_KURALS,
    MAX_GENERATION_LENGTH,
    NUM_RETURN_SEQUENCES,
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from gtts import gTTS

# --------------------------
# 🔠 Load Sentence-BERT for Embeddings
# --------------------------
print("🔠 Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# --------------------------
# 📚 Load and Prepare Thirukkural Dataset
# --------------------------
print("📖 Loading Thirukkural dataset...")
dataset = load_dataset("Selvakumarduraipandian/Thirukural")
sample_row = dataset["train"][0]
print("🔍 Sample row from dataset:", sample_row)
print("💑 Available keys:", list(sample_row.keys()))

kurals = [
    f"{re.sub('<.*?>', '', item.get('Kural', '').strip())}\n({re.sub('<.*?>', '', item.get('Couplet', '').strip())})"
    for item in dataset["train"]
    if item.get("Kural")
]

if not kurals:
    raise ValueError("❌ No valid 'Kural' entries found.")

# --------------------------
# 📌 Embed All Kurals
# --------------------------
print("📌 Generating embeddings for all Kurals...")
kural_embeddings = embedder.encode(kurals, convert_to_tensor=True).cpu().detach().numpy()
if kural_embeddings.shape[0] == 0:
    raise ValueError("❌ Kural embeddings are empty.")

# --------------------------
# 🧠 Setup Groq Client
# --------------------------
print("⚙️ Initializing Groq client...")
client = Groq(api_key=os.getenv("GROQ_API_KEY", "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"))

# --------------------------
# 🔍 Retrieve Top Kurals
# --------------------------
def retrieve_kurals(query: str, top_k: int = TOP_KURALS) -> list:
    if not query.strip():
        raise ValueError("❌ Empty query provided.")

    query_embedding = embedder.encode([query])
    print("✅ query_embedding shape:", query_embedding.shape)
    print("✅ kural_embeddings shape:", kural_embeddings.shape)

    scores = cosine_similarity(query_embedding, kural_embeddings)[0]
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [kurals[i] for i in top_indices]

# --------------------------
# 💬 Generate Advice & TTS Audio
# --------------------------
def generate_advice(dilemma: str):
    retrieved_kurals = retrieve_kurals(dilemma)

    # Format Kurals
    formatted_kurals = []
    for kural in retrieved_kurals:
        lines = kural.strip().split("\n")
        if len(lines) == 2:
            tamil = lines[0].strip()
            english = lines[1].strip()
            if len(tamil) > 40 and " " in tamil:
                split = tamil.find(" ", len(tamil) // 2)
                line1, line2 = tamil[:split], tamil[split + 1:]
                formatted_kurals.append(f"{line1}\n{line2}\n{english}")
            else:
                formatted_kurals.append(f"{tamil}\n{english}")
        else:
            formatted_kurals.append(kural.strip())

    kural_block = "\n\n".join(formatted_kurals)

    # Prompt for Groq model
    prompt = (
        f"You are a wise Tamil ethical advisor trained in the teachings of the Thirukkural.\n\n"
        f"The person is facing this ethical dilemma:\n'{dilemma}'\n\n"
        f"The following verses from the Thirukkural offer moral guidance:\n\n"
        f"{kural_block}\n\n"
        f"Based on these, provide compassionate, culturally respectful, and morally grounded advice.\n"
        f"Start your answer with: 'You should...'"
    )

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )

    reply = response.choices[0].message.content.strip()
    final_text = f"📜 Relevant Thirukkural(s):\n{kural_block}\n\n✅ AI Advice:\n{reply}"

    # TTS audio file generation
    tts = gTTS(final_text, lang='en')
    audio_path = "output_audio.mp3"
    tts.save(audio_path)

    return final_text, audio_path
