import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import shap

# ---------------- Load Pickle ----------------
@st.cache_resource
def load_model():
    with open("model3.pkl", "rb") as f:
        return pickle.load(f)

data = load_model()

# Extract objects from pickle
df = data["df"]
cosine_sim = data["cosine_sim"]
tfidf_vectorizer = data["tfidf_vectorizer"]
kmeans = data["kmeans"]
scaler = data["scaler"]
numeric_cols = data["numeric_cols"]
normalized = data["normalized"]

# ---------------- Helper Classes / Functions ----------------
class SpotifyRecommendation:
    def __init__(self, dataset, cosine_sim_matrix):
        self.dataset = dataset
        self.cosine_sim = cosine_sim_matrix

    def recommend(self, song_name, amount=5):
        indices = self.dataset.index[self.dataset['track_name'].str.lower() == song_name.lower()].tolist()
        if not indices:
            return None
        idx = indices[0]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:amount+1]
        track_indices = [i[0] for i in sim_scores]
        return self.dataset[['artists', 'track_name']].iloc[track_indices].reset_index(drop=True)

def get_cluster_recommendations(track_name, df_subset, amount=5):
    matches = df_subset[df_subset['track_name'].str.lower() == track_name.lower()]
    if matches.empty:
        return None
    track_cluster = int(matches['cluster'].iloc[0])
    cluster_df = df_subset[df_subset['cluster'] == track_cluster]
    if cluster_df.empty:
        return None
    return cluster_df[['artists', 'track_name']].sample(min(amount, len(cluster_df))).reset_index(drop=True)

def categorize_mood(valence):
    if valence > 0.7:
        return 'Positive'
    elif valence >= 0.4:
        return 'Neutral'
    else:
        return 'Negative'

if 'mood' not in df.columns:
    df['mood'] = df['valence'].apply(categorize_mood)

# ---------------- Streamlit UI ----------------
st.title("🎧 Spotify Recommendation System (Optimized + SHAP)")

menu = st.sidebar.selectbox("Choose a Feature", [
    "Content-Based Recommendation",
    "Cluster-Based Recommendation",
    "Mood-Based Recommendation",
    "Duration & Energy-Based Recommendation",
    "Visualizations",
    "Word Cloud",
    "Cluster Explainability (SHAP)"
])

# --- Content-Based ---
if menu == "Content-Based Recommendation":
    st.header("🎯 Content-Based Recommendation (Fast)")
    song_input = st.text_input("Enter a song name (any case):", "Hold On")
    num = st.slider("Number of recommendations", 1, 20, 5)
    if st.button("Recommend"):
        rec_engine = SpotifyRecommendation(df, cosine_sim)
        results = rec_engine.recommend(song_input, amount=num)
        if results is not None and not results.empty:
            st.dataframe(results)
            st.markdown("### 🤖 Why these songs were recommended")
            st.markdown(f"Based on TF-IDF and cosine similarity, we found songs most textually and acoustically similar to '{song_input}'. Features like artists, genre, danceability, energy, and valence were used to find similar tracks.")
        else:
            st.warning("Song not found in dataset. Try exact track name from your data.")

# --- Cluster-Based ---
elif menu == "Cluster-Based Recommendation":
    st.header("🧠 Cluster-Based Recommendation")
    song_input = st.text_input("Enter song for cluster recs:", "Hold On")
    num = st.slider("Number of recommendations", 1, 20, 5)
    if st.button("Recommend Cluster"):
        results = get_cluster_recommendations(song_input, df, amount=num)
        if results is not None and not results.empty:
            st.dataframe(results)
            st.markdown("### 🤖 Why these songs were recommended")
            st.markdown(f"The input song '{song_input}' belongs to a specific cluster based on its numeric audio features (e.g., energy, valence, danceability). The recommended songs are randomly sampled from the same cluster, sharing similar overall acoustic profiles.")
        else:
            st.warning("Song not found in dataset or no cluster matches.")

# --- Mood-Based ---
elif menu == "Mood-Based Recommendation":
    st.header("😊 Mood-Based Recommendation")
    mood = st.radio("Choose mood:", ['Positive', 'Neutral', 'Negative'])
    num = st.slider("Number of picks", 1, 20, 5)
    if st.button("Recommend by Mood"):
        sample = df[df['mood'] == mood].sample(min(num, len(df[df['mood'] == mood])))
        st.dataframe(sample[['artists', 'track_name', 'valence']].reset_index(drop=True))
        st.markdown("### 🤖 Why these songs were recommended")
        st.markdown(f"These songs fall under the '{mood}' category based on their valence scores — a measure of positivity in audio. Valence > 0.7 is Positive, 0.4–0.7 is Neutral, < 0.4 is Negative.")

# --- Duration & Energy Filter ---
elif menu == "Duration & Energy-Based Recommendation":
    st.header("⚡ Filter by Duration & Energy")
    dur = st.slider("Select Duration (ms)", int(df["duration_ms"].min()), int(df["duration_ms"].max()), (150000, 300000))
    energy = st.slider("Select Energy", 0.0, 1.0, (0.6, 1.0))
    num = st.slider("Number of recommendations", 1, 20, 5)
    if st.button("Recommend by Duration & Energy"):
        filtered = df[(df['duration_ms'].between(*dur)) & (df['energy'].between(*energy))]
        if filtered.empty:
            st.warning("No matches for chosen ranges.")
        else:
            st.dataframe(filtered[['artists', 'track_name', 'duration_ms', 'energy']].sample(min(num, len(filtered))).reset_index(drop=True))
            st.markdown("### 🤖 Why these songs were recommended")
            st.markdown("These songs were filtered based on your preferred **duration** and **energy** levels. Longer durations often align with more immersive listening, while higher energy indicates louder, faster-paced tracks. Songs within both thresholds were selected.")

# --- Visualizations ---
elif menu == "Visualizations":
    st.header("📊 Visual Analysis")
    with st.expander("Genre Popularity"):
        genre_pop = df.groupby("track_genre")["popularity"].mean().sort_values(ascending=False)
        st.bar_chart(genre_pop)

    with st.expander("Energy vs Loudness"):
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='energy', y='loudness', alpha=0.5, ax=ax)
        st.pyplot(fig)

    with st.expander("Danceability Distribution"):
        fig, ax = plt.subplots()
        sns.histplot(df["danceability"], bins=20, ax=ax)
        st.pyplot(fig)

    with st.expander("Acousticness by Genre (top 25 genres)"):
        top_genres = df['track_genre'].value_counts().nlargest(25).index
        fig, ax = plt.subplots(figsize=(10,5))
        sns.boxplot(x='track_genre', y='acousticness', data=df[df['track_genre'].isin(top_genres)], ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

# --- Word Cloud ---
elif menu == "Word Cloud":
    st.header("☁️ Word Cloud (interactive)")
    col_choice = st.selectbox("Choose column for word cloud", ['track_name', 'artists', 'album_name'])
    max_words = st.slider("Max words", 20, 500, 200)
    if st.button("Generate Word Cloud"):
        text = " ".join(df[col_choice].dropna().astype(str).tolist())
        if not text.strip():
            st.warning("No text found in chosen column.")
        else:
            wordcloud = WordCloud(width=800, height=400, max_words=max_words, background_color='white').generate(text)
            fig, ax = plt.subplots(figsize=(12,6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

# --- SHAP Explainability ---
elif menu == "Cluster Explainability (SHAP)":
    st.header("🔍 SHAP Explainability for Clusters (single-song)")
    st.info("Explains which numeric features pushed this song towards its cluster. Uses a small background sample for speed.")
    song_input = st.text_input("Enter a song name to explain:", "Hold On")
    if st.button("Explain Cluster (SHAP)"):
        indices = df.index[df['track_name'].str.lower() == song_input.lower()].tolist()
        if not indices:
            st.warning("Song not found in dataset. Use a track_name exactly as in your data.")
        else:
            idx = indices[0]
            background_size = min(50, normalized.shape[0])
            background = shap.sample(normalized, background_size, random_state=42)
            vector = normalized[idx].reshape(1, -1)
            cluster_assigned = int(kmeans.predict(vector)[0])
            with st.spinner("Computing SHAP values (may take 10–30s)..."):
                explainer = shap.KernelExplainer(kmeans.predict, background)
                shap_values = explainer.shap_values(vector)
            try:
                contrib = np.array(shap_values)[cluster_assigned][0]
            except Exception:
                contrib = np.array(shap_values)[0]
            contrib_series = pd.Series(contrib, index=numeric_cols).sort_values(key=lambda s: np.abs(s), ascending=True)
            fig, ax = plt.subplots(figsize=(8, max(4, 0.2*len(numeric_cols))))
            contrib_series.plot(kind='barh', ax=ax)
            ax.set_title(f"SHAP feature contributions for cluster {cluster_assigned} (song: {song_input})")
            st.pyplot(fig)
            st.write("Predicted cluster:", cluster_assigned)
            st.dataframe(df.loc[idx, numeric_cols].to_frame(name='value'))
