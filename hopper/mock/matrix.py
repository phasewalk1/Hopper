import attrs
import os
import numpy as np
import pandas as pd


@attrs.define
class MockMatrix:
    N_USERS: int
    N_SONGS: int
    TARGET_SPARSITY: float
    ALPHA: float
    BETA: float
    ACTIVITY_EXPONENT: float = 1.1
    POPULARITY_EXPONENT: float = 1.1

    N_TXS: int = attrs.field(init=False)

    # Instead, we set it here
    def __attrs_post_init__(self):
        self.N_TXS = int((self.TARGET_SPARSITY / 100) * (self.N_USERS * self.N_SONGS))

    def beta_distribution(self, seed=None, export=False, path=None) -> pd.DataFrame:
        if seed is not None:
            np.random.seed(seed)

        user_activity = (
            np.random.zipf(self.ACTIVITY_EXPONENT, self.N_TXS) % self.N_USERS + 1
        )

        item_popularity = (
            np.random.zipf(self.POPULARITY_EXPONENT, self.N_TXS) % self.N_SONGS + 1
        )

        ratings = np.random.beta(self.ALPHA, self.BETA, self.N_TXS) * 5.0
        ratings += (
            (user_activity + item_popularity) / (self.N_USERS + self.N_SONGS)
        ) * 0.5
        ratings = np.clip(ratings, 0, 5)

        data = {
            "user_id": user_activity.astype(np.int32),
            "song_id": item_popularity.astype(np.int32),
            "ratings": ratings,
        }

        df = pd.DataFrame(data)
        df = self.remove_dupls(df)

        if export is True:
            assert path is not None

            dir = os.path.dirname(path)
            if dir and not os.path.exists(dir):
                os.makedirs(dir)
            df.to_csv(path, index=False)

        return df

    def remove_dupls(self, matrix: pd.DataFrame) -> pd.DataFrame:
        matrix = matrix.drop_duplicates(subset=["user_id", "song_id"])
        return matrix
