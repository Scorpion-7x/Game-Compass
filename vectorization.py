# This file just like Data_Cleaner.py is just for processing the data one time and will not
# be used in user searching at all , this is just converted our cleaned_games.csv to matrix form for our model to process.
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

print("Initiating Game Compass AI Engine...")

# Checking if our model uses DGpu to run
if torch.cuda.is_available():
    device = 'cuda'
    print(f"Hardware selected: CUDA (GPU Detected!)")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    device = 'cpu'
    print("WARNING: GPU not detected. Running on CPU...")

#Loading the cleaned database as input
input_file = 'data/cleaned_games.csv'   

print("\nLoading cleaned dataset into memory for processing")
df = pd.read_csv(input_file)

#Processing the description to make it a string and make it into an array for pandas to work on
descriptions = df['description'].fillna("").astype(str).tolist()

print(f"Total games ready for the Game_Compass to process: {len(descriptions)}")

# Downloading the all-MiniLM model as Gpt suggested us to use this small 90 Mb model
print("\nLoading the AI Model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
print("The Model has been successfully loaded into your Computer.")

# Vectorizing the data as our model processes data into 384 dimensions matrix for it to work on

embeddings = model.encode(descriptions, batch_size=128, show_progress_bar=True)

# Printing out our matrix dimensions
print(f"Matrix shape: {embeddings.shape}")

# Getting the output file with our vectorized data
output_path = 'data/game_embeddings.npy'

print(f"\nLocation for Final Vectorized Data : {output_path}...Completed Vectorization!")
np.save(output_path, embeddings)