import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
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

    # Load the transaction matrix from a csv file
    def load(self, path: str):
        return pd.read_csv(
            path,
            index_col="user_id",
            usecols=["user_id", "song_id", "ratings"],
            dtype={"user_id": np.int32, "song_id": np.int32, "ratings": np.float32},
        )

    # Load the transaction matrix from a csv file and prune it
    #
    # Pruning operations:
    #   - Normalize interactions (min_interactions, max_interactions)
    #   - Prune low ratings (min_rating)
    @staticmethod
    def load_pruned(
        path: str,
        min_interactions: int,
        max_interactions: int,
        min_rating: float,
        against: pd.DataFrame = None,
    ):
        matrix = TxLoader.load(path)
        user_pruner, item_pruner = UserPruner(), ItemPruner()
        matrix = user_pruner.normalize_interactions(
            matrix, min_interactions, max_interactions
        )
        matrix = item_pruner.prune_low_ratings(matrix, min_rating)
        return matrix

    # Prune the tx matrix against the features. If a song in the tx matrix
    # does not contain any features, it is removed from both dataframes.
    @staticmethod
    def prune_against(
        matrix: pd.DataFrame,
        features: pd.DataFrame,
    ):
        matrix, features = ItemPruner.drop_songs_with_no_metadata(matrix, features)
        return matrix, features

    # Save the dataframe as a dataset for the model
    @staticmethod
    def save_set(matrix: pd.DataFrame, path: str):
        # user_id will be the index
        matrix.to_csv(path, columns=["song_id", "ratings"], mode="w")


# Loads the item metadata
class MetadataLoader(Dataloader):
    def __init__(self):
        super().__init__()

    def load(self, path: str):
        metadata = pd.read_csv(path, index_col="song_id")
        metadata = self.encode_cols(
            metadata, ["genre", "key", "majorOrMinor", "flatOrSharp"]
        )

        return metadata

    def encode_cols(self, metadata: pd.DataFrame, columns: list, replace=False):
        for col in columns:
            le = LabelEncoder()
            if replace:
                metadata[col] = le.fit_transform(metadata[col])
            else:
                metadata[col + "_encoded"] = le.fit_transform(metadata[col])
        return metadata
