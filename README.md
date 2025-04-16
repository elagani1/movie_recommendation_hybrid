# Hybrid Movie Recommendation System

This project combines fuzzy clustering and NLP to recommend movies:

## Features
- Fuzzy logic user clustering (MovieLens)
- Content-based filtering using TF-IDF (TMDB)
- Automatic Kaggle dataset download

## Instructions
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Place your `kaggle.json` in the proper `.kaggle/` folder.

3. Download MovieLens manually and extract into a folder named `ml-latest-small/`.

4. Run the app:
   ```
   python movie_recommendation_hybrid.py
   ```
