import pandas as pd
import streamlit as st
import pickle

# Load movie metadata and trained model
movies_info = pd.read_csv("movies_info.csv")
with open("Recommender.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Access internal item mapping from cmfrec
item_map = loaded_model.item_mapping  # external_id → internal_index
reverse_item_map = {v: k for k, v in item_map.items()}  # internal_index → external_id

# Streamlit UI
st.title("🎬 Zee Recommender System")
selected_titles = st.multiselect("Select movies you like", movies_info["Title"].tolist())

if st.button("Recommend Similar Movies"):
    try:
        # Map selected titles to external Movie IDs
        selected_external_ids = movies_info[movies_info["Title"].isin(selected_titles)]["Movie ID"].tolist()

        # Convert external IDs to internal indices used by cmfrec
        X_col = [item_map[movie_id] for movie_id in selected_external_ids if movie_id in item_map]
        X_val = [5.0] * len(X_col)  # Assume strong preference

        # Get top-N recommendations for this pseudo-user
        top_internal_ids = loaded_model.topN_warm(n=5, X_col=X_col, X_val=X_val)

        # Convert internal IDs back to external Movie IDs
        top_external_ids = [reverse_item_map[iid] for iid in top_internal_ids]

        # Filter out already selected movies
        recommended_ids = list(set(top_external_ids) - set(selected_external_ids))
        recommended_movies = movies_info[movies_info["Movie ID"].isin(recommended_ids)][["Title", "Genres"]]

        st.subheader("Recommended Movies Based on Your Selection:")
        st.table(recommended_movies)

    except Exception as e:
        st.error(f"Error: {e}")
