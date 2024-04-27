"""
This module is responsible for managing the data of the application.
"""
from typing import Iterator
from ucimlrepo import fetch_ucirepo


class DataManager:
    _DATABASES = {
        "wine_quality": 186,
        "iris": 61,
        "diabetes": 37,
        "boston": 13,
    }

    def __init__(self):
        self._cached_datasets = {}

    # =========== Static methods =========== #

    @staticmethod
    def get_data_from_file(fname: str) -> Iterator[str]:
        """
        Function to read the input file and return the data.
        """
        with open(fname, "r") as file_iter:
            for line in file_iter:
                line = line.strip().rstrip(",")  # Remove trailing comma
                record = frozenset(line.split(","))
                yield record

    # =========== Public methods =========== #

    def fetch_data(self, dataset: str):
        """
        Fetch the dataset from the UCI repository.

        Args:
            dataset (str): The name of the dataset.

        Returns:
            tuple: A tuple containing the features and targets of the dataset.
        """
        if dataset not in self._DATABASES.keys():
            raise ValueError(f"Dataset {dataset} not found.")

        if self._cached_datasets.get(dataset) is None:
            self._cached_datasets[dataset] = fetch_ucirepo(
                id=self._DATABASES[dataset]
            )

        return (
            self._cached_datasets[dataset].data.features,
            self._cached_datasets[dataset].data.targets
        )
