"""
data_processing.py
Loads and preprocesses the MovieLens 1M dataset for recommendation model training.
Outputs train/val/test splits and user/item feature matrices.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_movielens(ratings_path: str, movies_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    return ratings, movies


def encode_ids(ratings: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """Re-index user and movie IDs to contiguous integers."""
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    ratings = ratings.copy()
    ratings["user_idx"] = user_enc.fit_transform(ratings["user_id"])
    ratings["item_idx"] = item_enc.fit_transform(ratings["movie_id"])
    return ratings, user_enc, item_enc


def build_interaction_matrix(ratings: pd.DataFrame, n_users: int, n_items: int) -> csr_matrix:
    """Builds a sparse user-item interaction matrix (implicit feedback: rating > 0 = 1)."""
    data = np.ones(len(ratings))
    row = ratings["user_idx"].values
    col = ratings["item_idx"].values
    return csr_matrix((data, (row, col)), shape=(n_users, n_items))


def split_data(
    ratings: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(
        ratings, test_size=test_size, random_state=random_state, stratify=ratings["user_idx"] % 10
    )
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val, random_state=random_state
    )
    return train, val, test


def get_genre_features(movies: pd.DataFrame, item_enc: LabelEncoder) -> np.ndarray:
    """One-hot encode genres for each movie in item_enc order."""
    all_genres = sorted(
        {g for genres in movies["genres"].str.split("|") for g in genres}
    )
    genre_to_idx = {g: i for i, g in enumerate(all_genres)}

    # Only keep movies that appear in item_enc
    encoded_ids = item_enc.classes_  # original movie_ids in item order
    movie_map = movies.set_index("movie_id")["genres"].to_dict()

    features = np.zeros((len(encoded_ids), len(all_genres)), dtype=np.float32)
    for idx, mid in enumerate(encoded_ids):
        if mid in movie_map:
            for g in movie_map[mid].split("|"):
                if g in genre_to_idx:
                    features[idx, genre_to_idx[g]] = 1.0
    return features, all_genres
