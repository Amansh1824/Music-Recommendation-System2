# 🎧 Spotify Music Recommendation System (Streamlit)

This is an interactive **music recommendation system** built with [Streamlit](https://streamlit.io/) and powered by a pre-trained model stored in `model.pkl`.  
It replicates the full features of the original Jupyter Notebook, but loads instantly without retraining.

---

## 🚀 Features

### 1. Content-Based Recommendation  
- Uses **TF-IDF** and **cosine similarity** to find songs textually and acoustically similar to a given track.  
- Considers metadata and audio features like artist, album, genre, danceability, energy, valence, loudness, and more.

### 2. Cluster-Based Recommendation  
- Groups songs into clusters using **KMeans** on normalized numeric audio features.  
- Recommends songs from the same cluster as the chosen track.

### 3. Mood-Based Recommendation  
- Classifies songs into **Positive**, **Neutral**, or **Negative** based on **valence** score.  
- Allows quick mood-based song discovery.

### 4. Duration & Energy Filter  
- Filters tracks by user-selected **duration range** (ms) and **energy range** (0–1).

### 5. Visualizations  
- Genre popularity  
- Energy vs Loudness scatter plot  
- Danceability distribution histogram  
- Acousticness by genre (box plots)

### 6. Word Cloud  
- Generates a **word cloud** from track, artist, or album names.

### 7. SHAP Explainability  
- Explains which numeric features influenced a song's **cluster assignment** using **SHAP values**.

---

## 📂 Project Structure
music_recommender_app/
├── app.py # Streamlit app
├── model.pkl # Pre-trained models & dataset
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## 🛠️ Installation & Running Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd music_recommender_app

pip install -r requirements.txt

streamlit run app.py
