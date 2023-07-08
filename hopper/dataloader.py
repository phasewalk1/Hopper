import numpy as np
import pandas as pd
from pruner import ItemPruner, UserPruner
from sklearn.preprocessing import LabelEncoder


# Loads the user-item transaction matrix
class TxLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        return pd.read_csv(
            self.path,
            index_col="user_id",
            usecols=["user_id", "song_id", "ratings"],
            dtype={"user_id": np.int32, "song_id": np.int32, "ratings": np.float32},
        )

    # Load the transaction matrix from a csv file and prune it
    #
    # Pruning operations:
    #   - Normalize interactions (min_interactions, max_interactions)
    #   - Prune low ratings (min_rating)
    def load_pruned(
        self,
        min_interactions: int,
        max_interactions: int,
        min_rating: float,
    ):
        matrix = self.load()
        user_pruner, item_pruner = UserPruner(), ItemPruner()
        matrix = user_pruner.normalize_interactions(
            matrix,
            min_interactions,
            max_interactions,
        )
        matrix = item_pruner.prune_low_ratings(matrix, min_rating)
        return matrix

    # Prune the tx matrix against the features. If a song in the tx matrix
    # does not contain any features, it is removed from both dataframes.
    def prune_against(
        self,
        matrix: pd.DataFrame,
        features: pd.DataFrame,
    ):
        matrix, features = ItemPruner.drop_songs_with_no_metadata(matrix, features)
        return matrix, features

    # Save the dataframe as a dataset for the model
    def save_set(self, matrix: pd.DataFrame, path: str):
        # user_id will be the index
        matrix.to_csv(path, columns=["song_id", "ratings"], mode="w")


# Loads the item metadata
class MetadataLoader:
    def __init__(self, path):
        self.path = path

    def load(self, test=False):
        metadata = pd.read_csv(self.path, index_col="song_id")
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
