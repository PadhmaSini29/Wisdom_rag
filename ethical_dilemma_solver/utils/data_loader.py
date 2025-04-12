# data_loader.py

import os
import json
from datasets import load_dataset
from app.config import THIRUKKURAL_LOCAL_PATH



def load_thirukkural_data(use_local: bool = False) -> list:
    """
    Load Thirukkural English translations from Hugging Face or a local JSON file.

    Args:
        use_local (bool): If True and the file exists, loads dataset from a local JSON file.

    Returns:
        List[str]: A list of English translations of Thirukkural verses.
    """
    if use_local and os.path.exists(THIRUKKURAL_LOCAL_PATH):
        print("📁 Loading Thirukkural dataset from local file...")
        with open(THIRUKKURAL_LOCAL_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [item["Translation"] for item in data]

    print("🌐 Downloading Thirukkural dataset from Hugging Face...")
    dataset = load_dataset("Selvakumarduraipandian/Thirukural")
    train_data = dataset["train"]

    # Save locally for offline access
    print("💾 Saving a local backup of the dataset...")
    with open(THIRUKKURAL_LOCAL_PATH, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    # Return only the English translations (actual field is 'Translation')
    return [item["Translation"] for item in train_data]
