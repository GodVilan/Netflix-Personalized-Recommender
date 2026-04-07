"""
data_processing.py
Loads and preprocesses the MovieLens 1M dataset.

Key change: uses LEAVE-ONE-OUT split (standard ML-1M protocol).
  - Test set  = each user's LAST interaction (by timestamp)
  - Val set   = each user's SECOND-TO-LAST interaction
  - Train set = all remaining interactions
This prevents temporal leakage and matches published benchmark numbers.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder


def load_movielens(ratings_path: str, movies_path: str):
    ratings = pd.read_csv(
        ratings_path, sep="::", engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    movies = pd.read_csv(
        movies_path, sep="::", engine="python",
        names=["movie_id", "title", "genres"], encoding="latin-1",
    )
    return ratings, movies


def encode_ids(ratings: pd.DataFrame):
    """Re-index user and movie IDs to contiguous integers."""
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    ratings = ratings.copy()
    ratings["user_idx"] = user_enc.fit_transform(ratings["user_id"])
    ratings["item_idx"] = item_enc.fit_transform(ratings["movie_id"])
    return ratings, user_enc, item_enc


def build_interaction_matrix(ratings: pd.DataFrame, n_users: int, n_items: int) -> csr_matrix:
    """Sparse user-item implicit feedback matrix (any rating → 1)."""
    data = np.ones(len(ratings))
    row  = ratings["user_idx"].values
    col  = ratings["item_idx"].values
    return csr_matrix((data, (row, col)), shape=(n_users, n_items))


def split_data(ratings: pd.DataFrame):
    """
    Leave-One-Out (LOO) split — the standard ML-1M evaluation protocol.

    For each user:
      - Last interaction  (by timestamp) → test
      - Second-to-last   (by timestamp) → validation
      - Everything else  → train

    Why LOO instead of random split:
      Random splits leak future items into training, inflating train scores
      and deflating test NDCG by 2-3x vs. published baselines.
    """
    ratings = ratings.sort_values(["user_idx", "timestamp"])

    # Rank each interaction per user from oldest (1) to newest (n)
    ratings["rank"] = ratings.groupby("user_idx").cumcount(ascending=False)
    # rank=0 → last, rank=1 → second-to-last, rank>=2 → train

    test_df  = ratings[ratings["rank"] == 0].copy()
    val_df   = ratings[ratings["rank"] == 1].copy()
    train_df = ratings[ratings["rank"] >= 2].copy()

    return train_df, val_df, test_df


def get_genre_features(movies: pd.DataFrame, item_enc: LabelEncoder):
    """One-hot genre matrix (n_items × n_genres) in item_enc order."""
    all_genres = sorted(
        {g for genres in movies["genres"].str.split("|") for g in genres}
    )
    genre_to_idx = {g: i for i, g in enumerate(all_genres)}
    encoded_ids  = item_enc.classes_
    movie_map    = movies.set_index("movie_id")["genres"].to_dict()

    features = np.zeros((len(encoded_ids), len(all_genres)), dtype=np.float32)
    for idx, mid in enumerate(encoded_ids):
        if mid in movie_map:
            for g in movie_map[mid].split("|"):
                if g in genre_to_idx:
                    features[idx, genre_to_idx[g]] = 1.0
    return features, all_genres
