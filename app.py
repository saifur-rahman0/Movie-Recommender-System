import gradio as gr
import pandas as pd
import joblib

import os

# Load saved SVD model + movies
model_file = "svd_model_7866.pkl"
model = joblib.load(model_file) if os.path.exists(model_file) else None

movies_file = "movies.csv" if os.path.exists("movies.csv") else os.path.join("MovieLensDataset", "movie.csv")
if os.path.exists(movies_file):
    movies = pd.read_csv(movies_file)
else:
    movies = pd.DataFrame(columns=['movieId', 'title', 'genres'])

# Create mapping for quick lookup
movie_ids = movies['movieId'].tolist()

def recommend(user_id, top_k=10):
    if model is None:
        raise ValueError("Model file 'svd_model_7866.pkl' not found. Please train or place the model file in the directory.")
    preds = []
    for mid in movie_ids:
        pred = model.predict(user_id, mid).est
        preds.append((mid, pred))

    # Sort by predicted rating
    preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_k]
    top_movies = movies[movies['movieId'].isin([p[0] for p in preds])].copy()
    top_movies['predicted_rating'] = [p[1] for p in preds]
    return top_movies[['title', 'genres', 'predicted_rating']]

def gradio_recommender(user_id, top_k):
    try:
        user_id = int(user_id)
        recs = recommend(user_id, top_k=top_k)
        return recs.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}

demo = gr.Interface(
    fn=gradio_recommender,
    inputs=[
        gr.Textbox(label="Enter User ID"),
        gr.Slider(1, 20, value=10, step=1, label="Top-K Recommendations")
    ],
    outputs=gr.JSON(label="Top Recommendations"),
    title="🎬 Movie Recommender (SVD - Surprise)",
    description="Enter a user ID and choose Top-K to get personalized recommendations using SVD."
)

if __name__ == "__main__":
    demo.launch()
