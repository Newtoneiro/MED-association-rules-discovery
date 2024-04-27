import sys
from optparse import OptionParser
from src import Apriori, DataManager, AprioriUtils


if __name__ == "__main__":
    optparser = OptionParser()
    optparser.add_option(
        "-f",
        "--inputFile",
        dest="input",
        help="filename containing csv",
        default=None
    )
    optparser.add_option(
        "-s",
        "--minSupport",
        dest="minS",
        help="minimum support value",
        default=0.15,
        type="float",
    )
    optparser.add_option(
        "-c",
        "--minConfidence",
        dest="minC",
        help="minimum confidence value",
        default=0.6,
        type="float",
    )
    (options, args) = optparser.parse_args()

    inFile = None
    if options.input is None:
        inFile = sys.stdin
    elif options.input is not None:
        inFile = DataManager.get_data_from_file(options.input)
    else:
        print("No dataset filename specified, system with exit\n")
        sys.exit("System will exit")

    apriori = Apriori(options.minS, options.minC)

    items, rules = apriori.run(inFile)

    AprioriUtils.print_results(items, rules)