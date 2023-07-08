import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
import coloredlogs


def logging_environment():
    coloredlogs.install(level="DEBUG")


# Compute the Jaccard Similarity matrix for the given tx matrix
def jaccard_similarity(matrix: pd.DataFrame):
    dense = matrix.pivot(columns="song_id", values="ratings")
    jac_sim = 1 - pairwise_distances(dense.T.fillna(0), metric="hamming")
    return pd.DataFrame(jac_sim, index=dense.columns, columns=dense.columns)


# View the distribution of ratings
def view_rating_distribution(matrix: pd.DataFrame, show=True, save=False):
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.hist(matrix["ratings"], bins=5, rwidth=0.8, range=(0.5, 5.5))
    if show:
        plt.show()
    if save:
        plt.savefig("rating_distribution.png")
    plt.clf()
