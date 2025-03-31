import numpy as np
import ml_buckling as mlb
import pandas as pd
from mpi4py import MPI
import os, argparse, sys
import unittest
sys.path.append("../")
from _utils import get_metal_buckling_load, axial_load_old
import warnings

comm = MPI.COMM_WORLD

class TestAxial_Metal2Old(unittest.TestCase):

    def setUp(self):
        # Suppress the specific warning from scipy.optimize
        warnings.filterwarnings("ignore", category=UserWarning, message="delta_grad == 0.0.*")

    def test_old(self):

        eig_FEA, eig_CF, _, _ = axial_load_old(
            comm,
            rho0=0.6, # less than 0.6 fails..
            gamma=3.0,
            nstiff=5,
            prev_dict=None,
            solve_buckling=True
        )

        if comm.rank == 0:
            print(f"{eig_CF=} {eig_FEA=}")
            self.assertTrue(eig_FEA is not None)

if __name__ == "__main__":
    unittest.main()
