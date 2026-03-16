# coding=utf-8
from alg.algs.MILES import MILES

ALGORITHMS = [
    'MILES'
]


def get_algorithm_class(algorithm_name):
    if algorithm_name not in globals():
        raise NotImplementedError(
            "Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
