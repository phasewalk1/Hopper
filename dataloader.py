import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from pruner import UserPruner, ItemPruner


# A class that can handle preprocessing of data
class Dataloader(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def load(self, path: str):
        pass


# Loads the user-item transaction matrix
class TxLoader(Dataloader):
    def __init__(self):
        super().__init__()
        self.pruners = (UserPruner(), ItemPruner())

    def load(self, path: str):
        return pd.read_csv(
            path,
            index_col="user_id",
            usecols=["user_id", "song_id", "ratings"],
            dtype={"user_id": np.int32, "song_id": np.int32, "ratings": np.float32},
        )

    def prune(self, matrix: pd.DataFrame, min_interactions: int, max_interactions: int, min_rating: float):
        user_pruner, item_pruner = self.pruners
        matrix = user_pruner.normalize_interactions(matrix, min_interactions, max_interactions)
        matrix = item_pruner.prune_low_ratings(matrix, min_rating)
        return matrix
