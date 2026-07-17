import os
import math
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Checking our engine running hardware
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Booting Game Compass Search Engine on: {device.upper()}")

# Loading the data base from cleaned data and matrix from game_embeddings
csv_path = 'data/cleaned_games.csv'
matrix_path = 'data/game_embeddings.npy'

if not os.path.exists(csv_path) or not os.path.exists(matrix_path):
    print("❌ ERROR: Missing required data files. Ensure Phase 2 completed successfully.")
    exit()  

print("🧠 Loading vector space matrix and game database into memory...")
df = pd.read_csv(csv_path)
game_matrix = np.load(matrix_path)

# Load the NLP Transformer Model
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
print("✅ Search Engine Status: ONLINE & READY")

# --- STEP 3: The Hybrid Search Core Logic ---
def semantic_search(query, top_k=500):
    print(f"\n🔍 Processing natural language query: '{query}'")
    
    # 1. AI Vibe Check (Semantic Search)
    query_vector = model.encode(query, convert_to_tensor=True, device=device)
    matrix_tensor = torch.from_numpy(game_matrix).to(device)
    cos_scores = torch.nn.functional.cosine_similarity(query_vector, matrix_tensor, dim=1)
    
    # Grab a wide net of the top 500 mathematical matches
    top_results = torch.topk(cos_scores, k=top_k)
    
    print("⚖️ Applying hybrid ranking algorithm (Vibe + Popularity + Quality)...")
    
    # 2. Build a temporary ranking list
    candidates = []
    
    for score, index in zip(top_results.values, top_results.indices):
        game_idx = index.item()
        
        try:
            semantic_score = score.item()
            
            # 🚨 BULLETPROOF EXTRACTION USING INDEX POSITIONS 🚨
            title = str(df.iloc[game_idx, 0])            # Position 0: Title
            release_date = str(df.iloc[game_idx, 1])     # Position 1: Date
            description = str(df.iloc[game_idx, 2])      # Position 2: Description
            total_reviews = float(df.iloc[game_idx, 8])  # Position 8: Total Reviews
            positive_ratio = float(df.iloc[game_idx, 9]) # Position 9: Positive Ratio
            
            # Filter out the "shovelware" indie games with virtually no players
            if total_reviews < 200:
                continue
                
            # --- THE RANKING ALGORITHM ---
            # math.log10 scales down massive review counts so they don't break the math
            popularity_score = math.log10(total_reviews) / 7.0  
            
            # Weighted Composite Score: 60% Vibe, 20% Popularity, 20% Quality
            hybrid_score = (semantic_score * 0.60) + (popularity_score * 0.20) + (positive_ratio * 0.20)
            
            # Store the candidate
            candidates.append({
                'title': title,
                'release_date': release_date,
                'description': description,
                'semantic': semantic_score,
                'hybrid_score': hybrid_score,
                'reviews': int(total_reviews)
            })
            
        except (ValueError, ZeroDivisionError, TypeError):
            continue # If a specific row has corrupted numbers, skip it safely
            
    # 3. Sort by our new Hybrid Score and grab the Top 5
    candidates = sorted(candidates, key=lambda x: x['hybrid_score'], reverse=True)[:5]
    
    # 4. Display Results
    print("\n🎯 Ultimate Match Recommendations:")
    if not candidates:
        print("No practical matches found. Try a different description.")
        return
        
    for game in candidates:
        print(f"\n🎮 Title: {game['title']}")
        print(f"📊 Hybrid Score: {game['hybrid_score']:.4f} (AI Vibe: {game['semantic']:.4f} | Reviews: {game['reviews']})")
        print(f"📅 Released: {game['release_date']}")
        print(f"📝 Description snippet: {str(game['description'])[:140]}...")

# --- STEP 4: Interactive Search Loop ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🧭 GAME COMPASS TERMINAL ACTIVATED")
    print("="*50)
    
    while True:
        user_input = input("\nDescribe the game you want to play (or type 'exit' to quit): ")
        
        if user_input.lower() == 'exit':
            print("Shutting down engine. Goodbye!")
            break
            
        semantic_search(user_input, top_k=500)