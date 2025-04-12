# generation.py

from transformers import pipeline
from app.config import (
    GENERATION_MODEL_NAME,
    MAX_GENERATION_LENGTH,
    NUM_RETURN_SEQUENCES
)

# --------------------------
# Load Text Generation Model
# --------------------------
print("🧠 [generation.py] Loading generation model...")
generator = pipeline(
    "text-generation",
    model=GENERATION_MODEL_NAME,
    tokenizer=GENERATION_MODEL_NAME
)

# --------------------------
# Generate Advice
# --------------------------
def generate_advice_text(dilemma: str, kural_context: str) -> str:
    """
    Generate AI advice based on a user's ethical dilemma and retrieved Thirukkural verses.

    Args:
        dilemma (str): The user's ethical dilemma or question.
        kural_context (str): The combined text of relevant Thirukkural verses.

    Returns:
        str: AI-generated advice.
    """
    # Construct prompt with retrieval context
    prompt = (
        f"Based on the following Thirukkural principles:\n{kural_context}\n\n"
        f"How should one respond to this ethical dilemma: '{dilemma}'"
    )

    # Generate advice
    response = generator(
        prompt,
        max_length=MAX_GENERATION_LENGTH,
        num_return_sequences=NUM_RETURN_SEQUENCES,
        do_sample=True,
        temperature=0.8
    )

    # Return clean output
    return response[0]["generated_text"].strip()
