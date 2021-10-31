"""Argument parser and default parameters."""

import argparse

default_params = {
        'seed': 44,
        'algorithm': 'reg_logistic',
        'max_iters': 100,
        'gamma': 1e-4,
        'batch_size': 100,
        'k_folds': 10,
        'output_file': 'predictions'
    }


def parse_arguments():
    """
    Set arguments from the command line when running 'run.py'. Option '-h' or '--help' provides 
    information about parameters and its usage.

    returns:
    parser.parse_args() -- ArgumentParser with given arguments
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
        '--bound_outliers', action='store_true', help='Bound values of outliers', default=False
    )
    parser.add_argument(
        '--impute_median', action='store_true', help='Median imputing for unknown values', default=False
    )
    parser.add_argument(
        '--feature_expansion', action='store_true', help='Median imputing for unknown values', default=False
    )
    parser.add_argument(
        '--split_jet', action='store_true', help='Split data depending on the number of jets', default=False
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Provide additional details about the program. This level of detail'
             ' can be very helpful for troubleshooting problems', default=False
    )
    parser.add_argument(
        '--k_folds', type=int,
        help='Number of fold in cross validation'
    )
    parser.add_argument(
        '--max_iters', type=int,
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