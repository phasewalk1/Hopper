import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class Dataloader(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def load(self, path: str):
        pass


class TxLoader(Dataloader):
    def __init__(self):
        super().__init__()

    def load(self, path: str):
        return pd.read_csv(
            path,
            index_col="user_id",
            usecols=["user_id", "song_id", "ratings"],
            dtype={"user_id": np.int32, "song_id": np.int32, "ratings": np.float32},
        )
