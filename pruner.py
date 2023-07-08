import numpy as np


def hard_limit_prune(df, column, limit):
    return df[df[column] <= limit]


def random_prune(df, column, frac, seed):
    np.random.seed(seed)
    num_rows = len(df)
    rows_to_remove = int(num_rows * frac)
    random_indices = np.random.choice(num_rows, size=rows_to_remove, replace=False)
    return df.drop(df.index[random_indices])


class ItemPruner:
    def __init__(self):
        pass

    @staticmethod
    def prune_low_ratings(matrix, thresh):
        def filter(x):
            return x["rating"].max() >= thresh

        return matrix.groupby("song_id").filter(filter)

    @staticmethod
    def drop_songs_with_no_metadata(matrix, features):
        matrix = matrix[matrix["song_id"].isin(features.index)]
        features = features[features.index.isin(matrix["song_id"])]
        return matrix, features


class UserPruner:
    def normalize_interactions(self, matrix, min_interactions, max_interactions=None):
        if max_interactions is not None:
            matrix = hard_limit_prune(matrix, "user_id", max_interactions)
        matrix = hard_limit_prune(matrix, "user_id", min_interactions)
