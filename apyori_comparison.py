import itertools
from apyori import apriori
from src import Apriori, DataManager, Metrics, AprioriUtils
from itertools import tee


DATASETS = [
    "car_evaluation",
    # "tic_tac_toe_endgame",
    # "nursery"
]

SUPPORTS = [0.15]
CONFIDENCES = [0.6]


def main():
    data_manager = DataManager()

    for dataset, support, confidence in itertools.product(DATASETS, SUPPORTS, CONFIDENCES):
        my_apriori = Apriori(
            min_support=support,
            min_confidence=confidence,
        )

        data_x, data_y = data_manager.fetch_data_from_UCI(dataset)
        input = DataManager.combine_data(data_x, data_y)
        my_apriori_input, apyori_input = tee(input)

        items, rules = my_apriori.run(my_apriori_input)
        results = list(apriori(apyori_input))

        print(items[0], rules[0])
        print(results[0])


if __name__ == "__main__":
    main()
