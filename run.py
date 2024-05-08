import sys
import click
from src import Apriori, DataManager, AprioriUtils, Metrics
from itertools import tee


@click.command()
@click.option(
    "-f",
    "--input-file",
    required=False,
    type=click.Path(exists=True),
    help="Input file containing the dataset.",
)
@click.option(
    "-u",
    "--UCI-dataset",
    required=False,
    type=str,
    default="car_evaluation",
    help="UCI dataset name.",
)
@click.option(
    "-s",
    "--min-support",
    required=False,
    type=float,
    default=0.15,
    help="Minimum support value.",
)
@click.option(
    "-c",
    "--min-confidence",
    required=False,
    type=float,
    default=0.6,
    help="Minimum confidence value.",
)
def main(
        input_file: click.Path,
        uci_dataset: str,
        min_support: float,
        min_confidence: float
        ):
    """
    Main function to run the Apriori algorithm.

    :param input_file: Input file containing the dataset.\n
    :param uci_dataset: UCI dataset name.\n
    :param min_support: Minimum support value.\n
    :param min_confidence: Minimum confidence value.\n
    """

    data_manager = DataManager()
    if input_file is not None:
        input = DataManager.get_data_from_file(input_file)
    elif uci_dataset is not None:
        data_x, data_y = data_manager.fetch_data_from_UCI(uci_dataset)
        input = DataManager.combine_data(data_x, data_y)
    else:
        print("No dataset filename specified\n")
        sys.exit(0)

    apriori = Apriori(min_support, min_confidence)

    apriori_input, metrics_input = tee(input)

    items, rules = apriori.run(apriori_input)
    metrics = Metrics(metrics_input, items)
    rules_metrics = metrics.get_metrics(rules)

    AprioriUtils.print_results(items, rules)
    AprioriUtils.print_metrics(rules_metrics)


if __name__ == "__main__":
    main()
