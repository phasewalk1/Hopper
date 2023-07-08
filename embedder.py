import pandas as pd
import numpy as np
from tqdm import tqdm


def embed_one(matrix_slice: pd.DataFrame, features_df: pd.DataFrame) -> np.ndarray:
    if "ratings" not in matrix_slice.columns or "song_id" not in matrix_slice.columns:
        raise ValueError("Matrix slice must have ratings and song_id columns")

    avg_rating = matrix_slice["ratings"].mean()
    neutral_rating = 2.5
    biased_avg_rating = (avg_rating + neutral_rating) / 2
    normalized_ratings: np.ndarray = (
        matrix_slice["ratings"] - biased_avg_rating
    ).values.reshape(-1, 1)
    item_metadata = features_df.loc[matrix_slice["song_id"].values].values
    user_embedding = np.dot(normalized_ratings.T, item_metadata)
    user_embedding = user_embedding.mean(axis=0)

    return user_embedding


def embed_many(matrix: pd.DataFrame, features_df: pd.DataFrame):
    index = matrix.index.unique().copy()
    data = np.zeros((len(index), features_df.shape[1]))
    user_embeddings = pd.DataFrame(index=index, data=data)

    for user_id, user_ratings in tqdm(
        matrix.groupby("user_id"), desc="Creating user embeddings"
    ):
        slice = user_ratings.iloc[0]
        user_embeddings.loc[user_id, :] = embed_one(slice, features_df)

    return user_embeddings
