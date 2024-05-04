import sys
from argparse import ArgumentParser
from src import Apriori, DataManager, AprioriUtils, Metrics
from itertools import tee


if __name__ == "__main__":
    optparser = ArgumentParser()
    optparser.add_argument(
        "-f",
        "--inputFile",
        dest="input",
        help="filename containing csv",
        default=None
    )
    optparser.add_argument(
        "-u",
        "--UCIDataset",
        dest="UCI",
        help="fetch data from UCI repository",
        default="car_evaluation"
    )
    optparser.add_argument(
        "-s",
        "--minSupport",
        dest="minS",
        help="minimum support value",
        default=0.15,
        type=float
    )
    optparser.add_argument(
        "-c",
        "--minConfidence",
        dest="minC",
        help="minimum confidence value",
        default=0.6,
        type=float,
    )
    options = optparser.parse_args()

    data_manager = DataManager()
    if options.input is not None:
        input = DataManager.get_data_from_file(options.input)
    elif options.UCI is not None:
        data_x, data_y = data_manager.fetch_data_from_UCI(options.UCI)
        input = DataManager.combine_data(data_x, data_y)
    else:
        print("No dataset filename specified\n")
        sys.exit(0)

    apriori = Apriori(options.minS, options.minC)

    apriori_input, metrics_input = tee(input)

    items, rules = apriori.run(apriori_input)
    metrics = Metrics(metrics_input, items)
    rules_metrics = metrics.get_metrics(rules)

    AprioriUtils.print_results(items, rules)
    AprioriUtils.print_metrics(rules_metrics)
