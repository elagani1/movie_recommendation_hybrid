import os
import zipfile
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import skfuzzy as fuzz
import subprocess

pd.read_csv("tmdb_movies_data.csv")

warnings.filterwarnings("ignore")

### === AUTOMATED TMDB DATASET SETUP === ###

def download_tmdb_dataset():
    kaggle_dataset = 'asaniczka/tmdb-movies-dataset-2023-930k-movies'
    zip_path = 'tmdb-movies.zip'
    csv_filename = 'tmdb_movies_data.csv'

    if not os.path.exists(csv_filename):
        print("Downloading TMDB dataset from Kaggle...")
        subprocess.run(['kaggle', 'datasets', 'download', '-d', kaggle_dataset, '-f', csv_filename], check=True)

        print("Unzipping dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall()

        print(f"{csv_filename} extracted and ready.\n")
    else:
        print(f"{csv_filename} already exists. Skipping download.\n")

### === PART 1: Fuzzy Clustering (MovieLens) === ###

def load_movielens_data():
    ratings = pd.read_csv("ml-latest-small/ratings.csv")
    movies = pd.read_csv("ml-latest-small/movies.csv")
    df = pd.merge(ratings, movies, on='movieId')
    user_movie_matrix = df.pivot_table(index='userId', columns='title', values='rating').fillna(0)
    scaler = MinMaxScaler()
    user_movie_matrix_scaled = scaler.fit_transform(user_movie_matrix)
    return df, user_movie_matrix, user_movie_matrix_scaled, scaler

def fuzzy_cluster_users(user_movie_matrix_scaled, n_clusters=5):
    data = user_movie_matrix_scaled.T
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None)
    return cntr, u

def recommend_from_cluster(user_ratings, user_movie_matrix, scaler, cntr, u):
    new_user_vector = np.zeros(user_movie_matrix.shape[1])
    for idx, title in enumerate(user_movie_matrix.columns):
        if title in user_ratings:
            new_user_vector[idx] = user_ratings[title]

    new_user_vector_scaled = scaler.transform([new_user_vector])
    _, u_new, _, _, _, _, _ = fuzz.cluster.cmeans_predict(new_user_vector_scaled.T, cntr, m=2, error=0.005, maxiter=1000)
    top_cluster = np.argmax(u_new)
    cluster_users = np.where(u[top_cluster] > 0.5)[0]
    top_users = user_movie_matrix.iloc[cluster_users]
    top_movies = top_users.mean().sort_values(ascending=False).head(10)
    return top_movies

### === PART 2: NLP-based Content Filtering (TMDB) === ###

def load_tmdb_data():
    df = pd.read_csv("tmdb_movies_data.csv")
    df = df[['title', 'overview', 'genres']]
    df.dropna(subset=['title', 'overview'], inplace=True)
    df['title'] = df['title'].str.lower()
    df['overview'] = df['overview'].str.lower()
    df = df.drop_duplicates(subset='title')
    return df

def build_tfidf_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    return tfidf_matrix

def recommend_by_overview(user_input_title, df, tfidf_matrix, top_n=5):
    all_titles = df['title'].tolist()
    matched_title, score = process.extractOne(user_input_title.lower(), all_titles)
    idx = df[df['title'] == matched_title].index[0]
    sim_scores = list(enumerate(cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return matched_title, df.iloc[movie_indices][['title', 'genres', 'overview']]

### === PART 3: Unified Interface === ###

def hybrid_recommender():
    print("\n=== Starting Movie Recommendation System ===")
    
    # Download TMDB dataset if needed
    download_tmdb_dataset()

    # Load datasets
    print("Loading MovieLens data...")
    ml_df, user_movie_matrix, user_movie_matrix_scaled, scaler = load_movielens_data()
    
    print("Clustering users using fuzzy logic...")
    cntr, u = fuzzy_cluster_users(user_movie_matrix_scaled)
    
    print("Loading TMDB movie data...")
    tmdb_df = load_tmdb_data()
    tfidf_matrix = build_tfidf_matrix(tmdb_df)

    # User Input
    print("\nEnter a movie you like (for NLP-based recommendation):")
    user_movie = input("Movie Title: ")

    user_ratings = {
        "Toy Story (1995)": 5,
        "Jumanji (1995)": 3,
        "Grumpier Old Men (1995)": 4,
        "Heat (1995)": 2
    }

    print("\n=== Top NLP-based Recommendations ===")
    matched_title, content_recs = recommend_by_overview(user_movie, tmdb_df, tfidf_matrix)
    print(f"\nYour input matched: {matched_title.title()}\n")
    print(content_recs.to_string(index=False))

    print("\n=== Top Fuzzy Cluster-based Recommendations ===")
    cluster_recs = recommend_from_cluster(user_ratings, user_movie_matrix, scaler, cntr, u)
    print(cluster_recs.to_string())

if __name__ == '__main__':
    hybrid_recommender()
