"""
The Unified Bigram Generator
Copy this into a cell or a .py script and run it. It will automatically create a dummy text file, train the model, and generate a few sample sentences.
"""

import random
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple

# 1. DATACLASSES & TYPE HINTS
# Structured storage for our model's metadata.
@dataclass
class BigramModelStats:
    filename: str
    vocab_size: int
    total_transitions: int

def train_bigram_model(filepath: str) -> Tuple[Dict[str, Counter], BigramModelStats]:
    """Reads a file, pads the text, and builds a bigram transition matrix."""
    # 2. CONTEXT MANAGER (File Handling)
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    # 3. LIST COMPREHENSION & CONCATENATION (Padding)
    # Split text into words, strip whitespace, and add our special START and END tokens
    words: List[str] = [w.strip() for w in text.split()]
    words = ["<S>"] + words + ["</S>"]
    
    # 4. DEFAULTDICT & COUNTER
    # Automatically handles missing keys by creating an empty Counter
    bigram_counts = defaultdict(Counter)
    
    # 5. SLIDING WINDOW ITERATION
    # Iterate through overlapping pairs to build transition frequencies
    for w1, w2 in zip(words, words[1:]):
        bigram_counts[w1][w2] += 1
        
    # Compile stats to return alongside the model
    total_trans = sum(sum(counts.values()) for counts in bigram_counts.values())
    stats = BigramModelStats(filepath, len(bigram_counts), total_trans)
    
    return bigram_counts, stats

def generate_sentence(bigram_model: Dict[str, Counter], max_words: int = 15) -> str:
    """Generates a sentence by walking the bigram transition matrix."""
    current_word = "<S>"
    generated_words: List[str] = []
    
    for _ in range(max_words):
        transitions = bigram_model[current_word]
        vocab = list(transitions.keys())
        
        # 6. TRUTHINESS (Reactive safety check)
        if not vocab:
            break
            
        frequencies = list(transitions.values())
        
        # 7. WEIGHTED SAMPLING
        next_word = random.choices(vocab, weights=frequencies, k=1)[0]
        
        # 8. ORGANIC ENDING (Proactive safety check)
        if next_word == "</S>":
            break
            
        generated_words.append(next_word)
        
        # Move the window forward
        current_word = next_word
        
    return " ".join(generated_words)

# ==========================================
# Execution Pipeline
# ==========================================
if __name__ == "__main__":
    # Create some training data
    sample_data = "apple banana apple orange banana apple grape apple orange"
    with open("training_data.txt", "w", encoding="utf-8") as f:
        f.write(sample_data)
        
    # Train the model
    model, stats = train_bigram_model("training_data.txt")
    print(f"--- Model Training Complete ---")
    print(f"Stats: {stats}\n")
    
    # Generate 5 sample sentences
    print("--- Generated Sentences ---")
    for i in range(5):
        sentence = generate_sentence(model)
        print(f"Generation {i+1}: {sentence}")
