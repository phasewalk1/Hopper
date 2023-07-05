import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from abc import ABC, abstractmethod

from dataloader import Dataloader, TxLoader
from globals import MOCK_MATRIX_FILE


# Compute the Jaccard Similarity matrix for the given tx matrix
def jaccard_similarity(matrix: pd.DataFrame):
    dense = matrix.pivot(columns='song_id', values='ratings')
    jac_sim = 1 - pairwise_distances(dense.T.fillna(0), metric="hamming")
    return pd.DataFrame(jac_sim, index=dense.columns, columns=dense.columns)


# A View is a helper class for data visualization and analysis
class View(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def view(self, data: pd.DataFrame):
        pass


# A View for the transaction matrix
class TxMatrixView(View):
    def __init__(self):
        super().__init__()

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
