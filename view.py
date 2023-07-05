import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from dataloader import Dataloader, TxLoader
from globals import MOCK_MATRIX_FILE


class View(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def view(self, data: pd.DataFrame):
        pass


class TxMatrixView(View):
    def __init__(self):
        super().__init__()

    # 'user_id', 'song_id', 'ratings'
    def view(self, matrix: pd.DataFrame, show=True, save=False):
        plt.title("Rating Distribution")
        plt.xlabel("Rating")
        plt.ylabel("Count")
        plt.hist(matrix["ratings"], bins=5, rwidth=0.8, range=(0.5, 5.5))
        if show:
            plt.show()
        if save:
            plt.savefig("rating_distribution.png")
        plt.clf()


if __name__ == "__main__":
    loader: Dataloader = TxLoader()
    view: View = TxMatrixView()
    data: pd.DataFrame = loader.load(MOCK_MATRIX_FILE)
    view.view(data, show=True, save=False)
