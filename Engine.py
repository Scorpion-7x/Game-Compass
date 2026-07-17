import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler

# Setting up the Main Heading
st.set_page_config(page_title="Game Compass", page_icon="🧭", layout="centered")


#Loading the Engine
@st.cache_resource
def load_ai_engine():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    return model, device

@st.cache_data
def load_database():
    csv_path = 'data/cleaned_games.csv'
    matrix_path = 'data/game_embeddings.npy'
    
    if not os.path.exists(csv_path) or not os.path.exists(matrix_path):
        st.error("Error(420) : Missing required data files.")
        st.stop()
        
    df = pd.read_csv(csv_path)
    game_matrix = np.load(matrix_path)
    return df, game_matrix

# Initialize Backend
model, device = load_ai_engine()
df, game_matrix = load_database()

# Searching logic inspired by search.py file and incorporated with frontend
def get_semantic_matches(query, top_k=500):
    query_vector = model.encode(query, convert_to_tensor=True, device=device)
    matrix_tensor = torch.from_numpy(game_matrix).to(device)
    cos_scores = torch.nn.functional.cosine_similarity(query_vector, matrix_tensor, dim=1)
    
    top_results = torch.topk(cos_scores, k=top_k)
    
    candidates = []
    for score, index in zip(top_results.values, top_results.indices):
        game_idx = index.item()
        
        try:
            total_reviews = float(df.iloc[game_idx, 8])
            # Base quality filter
            if total_reviews < 200:
                continue
                
            candidates.append({
                'Game Title': str(df.iloc[game_idx, 0]),
                'Release Date': str(df.iloc[game_idx, 1]),
                'Description': str(df.iloc[game_idx, 2]),
                'total_reviews': total_reviews,
                'positive_ratio': float(df.iloc[game_idx, 9]),
                'semantic_score': score.item()
            })
        except:
            continue
            
    return pd.DataFrame(candidates)

# Header UI  (Main Title + Second line)
st.markdown("""
    <div style="text-align: center; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; padding-bottom: 20px;">
        <h1 style="font-weight: 700; letter-spacing: -1.5px; font-size: 3rem;">Game Compass</h1>
        <h3 style="color: #888888; font-weight: 400;">Find your next gaming adventure.</h3>
    </div>
""", unsafe_allow_html=True)

# Interactive UI (Search Bar + Ratio sliders)
user_query = st.text_input("", placeholder="Describe the vibe or gameplay experience you want...")
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("#### Adjust the Engine")
col1, col2 = st.columns(2)
with col1:
    volume_trust_weight = st.slider("Community Trust Ratio :", 0.0, 1.0, 0.55)
with col2:
    semantic_weight = st.slider("Semantic Vibe Ratio :", 0.0, 1.0, 0.80)

st.divider()

# Scores and calculation via the weightage and from the database (game_embeddings.npy)
def normalize_and_score(results_df):
    if results_df.empty:
        return results_df
        
    scaler = MinMaxScaler()
    results_df[['norm_pos', 'norm_vol']] = scaler.fit_transform(results_df[['positive_ratio', 'total_reviews']])
    
    # Base weight for quality
    quality_weight = 0.5 
    
    # Calculate total weight from the three ways
    total_weight = semantic_weight + volume_trust_weight + quality_weight
    
    # Calculating normalized weights
    w_sem = semantic_weight / total_weight
    w_vol = volume_trust_weight / total_weight
    w_qual = quality_weight / total_weight
    
    results_df['final_score'] = (
        (results_df['semantic_score'] * w_sem) + 
        (results_df['norm_pos'] * w_qual) + 
        (results_df['norm_vol'] * w_vol)
    )
    return results_df.sort_values(by='final_score', ascending=False).head(5)

# Search engine execution with output format
if st.button("Discover Games", use_container_width=True):
    if user_query:
        with st.spinner('Powering up the Graphical Processing unit to search vector space..'):
            # Fetch raw math matches
            raw_matches = get_semantic_matches(user_query)
            
            if raw_matches.empty:
                st.warning("Sorry, No highly-reviewed games matched this vibe. Try a broader search.")
            else:
                # UI normalization for results
                final_results = normalize_and_score(raw_matches)
                
                st.success("Here are your top matches :-")
                
                # Format for Result's Display
                for _, row in final_results.iterrows():
                    with st.container():
                        st.subheader(f"🎮{row['Game Title']}")
                        st.caption(f"Released: {row['Release Date']} | Match Score: {row['final_score']:.3f} | Reviews: {int(row['total_reviews'])}")
                        st.write(str(row['Description'])[:250] + "...")
                        st.divider()
    else:
        st.warning("Please tell us what you're looking for first.:)")
