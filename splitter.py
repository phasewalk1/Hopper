import pandas as pd
from globals import SEED


class Splitter:
    @staticmethod
    def split_flat(
        matrix: pd.DataFrame, val_size: float, test_size: float, seed: int = SEED
    ):
        np.random.seed(seed)

        size = len(matrix)
        val_split = int(np.floor(size * val_size))
        test_split = int(np.floor(size * test_size))

        indices = list(range(size))
        np.random.shuffle(indices)

        val = matrix.iloc[indices[:val_split]]
        test = matrix.iloc[indices[val_split:test_split]]
        train = matrix.iloc[indices[test_split:]]

        return train, val, test
