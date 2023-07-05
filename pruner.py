import pandas as pd
import numpy as np


# Generic pruning operations
class Pruner:
    def __init__(self):
        pass

    def hard_limit_prune(self, df, column, limit):
        return df[df[column] <= limit]

    def random_prune(self, df, column, frac, seed):
        np.random.seed(seed)
        num_rows = len(df)
        rows_to_remove = int(num_rows * frac)
        random_indices = np.random.choice(num_rows, size=rows_to_remove, replace=False)
        return df.drop(df.index[random_indices])


# Pruning operations on items
class ItemPruner:
    def __init__(self):
        pass

    def prune_low_ratings(self, matrix, thresh):
        func = lambda x: x["rating"].max() >= thresh
        return matrix.groupby("song_id").filter(func)


# Pruning operations on users
class UserPruner(Pruner):
    def __init__(self):
        super().__init__()

    def normalize_interactions(self, matrix, min_interactions, max_interactions=None):
        if max_interactions is not None:
            matrix = self.hard_limit_prune(matrix, "user_id", max_interactions)
        matrix = self.hard_limit_prune(matrix, "user_id", min_interactions)
