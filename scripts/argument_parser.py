"""Argument parser and default parameters."""

import argparse

default_params = {
        'seed': 44,
        'algorithm': 'reg_logistic',
        'max_iters': 5000,
        'gamma': 1e-5,
        'batch_size': 100,
        'kfolds': 10,
        'output_file': 'predictions'
    }


def parse_arguments():
    """
    Set arguments from the command line when running 'run.py'. Option '-h' or '--help' provides 
    information about parameters and its usage.
    :return: parser.parse_args()
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS
    )

    parser.add_argument(
        '--algorithm',
        help='Algorithm for making predictions'
    )

    parser.add_argument(
        '--output_file',
        help='Output file name'
    )

    parser.add_argument(
        '--outliers', action='store_true', help='Filter outliers', default=False
    )
    parser.add_argument(
        '--median_imputing', action='store_true', help='Median imputing for unknown values', default=False
    )
    parser.add_argument(
        '--feature_expansion', action='store_true', help='Median imputing for unknown values', default=False
    )
    parser.add_argument(
        '--split_jet', action='store_false', help='Split data depending on the number of jets', default=True
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Provide additional details about the program. This level of detail'
             ' can be very helpful for troubleshooting problems', default=False
    )
    parser.add_argument(
        '--kfolds', type=int,
        help='Number of fold in cross validation'
    )
    parser.add_argument(
        '--max_iterations', type=int,
        help='Maximum number of iterations of the (stohastic) gradient descent algorithm'
    )
    parser.add_argument(
        '--batch_size', type=int,
        help='Number of instances in each batch during training'
    )
    parser.add_argument(
        '--gamma', type=float,
        help='Learning rate of the (stohastic) gradient descent algorithm'
    )
    parser.set_defaults(**default_params)

    return parser.parse_args()