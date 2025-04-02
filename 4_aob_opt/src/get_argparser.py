import numpy as np
import argparse

def get_argparser():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--procs", type=int, default=4)
    parent_parser.add_argument("--ksWeight", type=np.double, default=100.0)

    init_true_options = ['useML']
    init_false_options = ['hotstart', 'coldstart', 'struct_output', 'plot_hist', 'test_derivs']
    for option in init_true_options:
        parent_parser.add_argument(
            f"--{option}", default=True, action=argparse.BooleanOptionalAction
        )
    for option in init_false_options:
        parent_parser.add_argument(
            f"--{option}", default=False, action=argparse.BooleanOptionalAction
        )
    return parent_parser