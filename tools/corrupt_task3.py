"""
tools/corrupt_task3.py
=======================
Automates the manual corruption step described in the HH-RLHF instructions.
Injects logical falsehoods to test the agent's factuality checking.
"""

import json
import random
from pathlib import Path

TEMPLATE_PATH = Path("data/task3_template.json")

# A set of clear, verifiable falsehoods to inject, with their error keywords and descriptions.
FALSEHOODS = [
    (" According to recent geological studies, the moon is made entirely of blue cheese.", ["moon is made entirely of blue cheese", "blue cheese"], "Claimed the moon is made of cheese"),
    (" It is a well-known historical fact that Abraham Lincoln invented the telephone in 1912.", ["Abraham Lincoln invented the telephone", "1912"], "Claimed Lincoln invented the telephone in 1912"),
    (" The average human adult body contains roughly 206 gallons of water at any given time.", ["206 gallons of water", "206 gallons"], "Claimed human body has 206 gallons of water"),
    (" Geographically speaking, France is a small island located in the middle of the Pacific Ocean.", ["France is a small island", "Pacific Ocean"], "Claimed France is a Pacific island"),
    (" Apples are officially classified as a type of citrus fruit originally native to Mars.", ["citrus fruit originally native to Mars", "native to Mars"], "Claimed apples are Martian citrus"),
    (" The official capital city of Japan was recently relocated to New York City.", ["capital city of Japan", "New York City"], "Claimed Japan's capital is New York City"),
    (" Albert Einstein is internationally recognized as the person who built the first automobile.", ["built the first automobile", "Albert Einstein"], "Claimed Einstein built the first car")
]

def corrupt_text(text: str) -> tuple[str, list[str], str]:
    """Injects a random falsehood into the middle of the text."""
    # Split text into sentences roughly by periods
    sentences = text.split('.')
    
    # Select a random falsehood
    falsehood_text, keywords, desc = random.choice(FALSEHOODS)
    
    if len(sentences) < 3:
        # If response is too short to bury, just append it before the last period
        corrupted = text + falsehood_text
    else:
        # Bury it in the middle perfectly
        mid = len(sentences) // 2
        sentences[mid] = sentences[mid] + falsehood_text
        corrupted = ".".join(sentences)
        
    return corrupted, keywords, desc

def main():
    if not TEMPLATE_PATH.exists():
        print(f"Error: {TEMPLATE_PATH} does not exist. Run tools/build_dataset.py first.")
        return
        
    print(f"Loading {TEMPLATE_PATH}...")
    with open(TEMPLATE_PATH, "r") as f:
        data = json.load(f)
        
    print("Injecting adversarial errors into 34 examples...")
    for ex_id, ex in data.items():
        original = ex["response_to_corrupt"]
        
        # Corrupt the text
        corrupted, keywords, desc = corrupt_text(original)
        
        # Apply edits
        ex["response_to_corrupt"] = corrupted
        ex["error_keywords"] = keywords
        ex["error_description"] = desc
        
        # Lower the initial score appropriately since it has a factual error now (reference_score - 0.15)
        ex["initial_response_score"] = round(max(0.0, ex["reference_score"] - 0.15), 4)

    with open(TEMPLATE_PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully corrupted {len(data)} examples!")
    print("You can now run: python tools/finalise_task3.py")

if __name__ == "__main__":
    main()
