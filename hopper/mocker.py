from globals import SEED
import click
import logging
import os

from mock.matrix import MockMatrix
from view import logging_environment


@click.command()
@click.option("--export", is_flag=True, default=False, help="Export the mock tx matrix")
@click.option("--num-users", default=10000, type=int, help="User size")
@click.option("--num-songs", default=100000, type=int, help="Song size")
def mock_tx_matrix_with_beta_distribution(
    num_users=10000,
    num_songs=100000,
    export=False,
):
    logging_environment()
    if not export:
        logging.info(
            "export=False: Run with 'python mocker.py --export' to export the mock matrix"
        )
    else:
        logging.info("export=True: Exporting the mock matrix to ../data/mock/")
    logging.debug(f"num_users={num_users}")
    logging.debug(f"num_songs={num_songs}")

    N_USERS = num_users
    N_SONGS = num_songs
    SPARSITY = 0.2
    ALPHA = 4.5
    BETA = 5.0

    logging.debug(f"alpha={ALPHA}")
    logging.debug(f"beta={BETA}")

    mocker = MockMatrix(
        N_USERS,
        N_SONGS,
        SPARSITY,
        ALPHA,
        BETA,
    )

    if export is True:
        path = os.path.join(os.path.dirname(__file__), "../data/mock/mock-matrix.csv")
        return mocker.beta_distribution(
            seed=SEED, export=True, path=path
        )
    else:
        return mocker.beta_distribution(seed=SEED)


if __name__ == "__main__":
    mock_tx_matrix_with_beta_distribution()
