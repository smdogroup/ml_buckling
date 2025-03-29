import numpy as np
import ml_buckling as mlb
import pandas as pd
from mpi4py import MPI
import os, argparse, sys
import unittest
from _utils import get_metal_buckling_load
import warnings

comm = MPI.COMM_WORLD

class TestAxial_Metal1(unittest.TestCase):

    def setUp(self):
        # Suppress the specific warning from scipy.optimize
        warnings.filterwarnings("ignore", category=UserWarning, message="delta_grad == 0.0.*")
    
    def test_case1(self):

        eig_CF, eig_FEA = get_metal_buckling_load(
            comm,
            rho0=1.0,
            gamma=3.0,
            num_stiff=1,
            sigma_eig=5.0,
            stiff_AR=20.0,
            plate_slenderness=100.0,
            is_axial=True,
        )

        if comm.rank == 0:
            print(f"{eig_CF=} {eig_FEA=}")
            self.assertTrue(eig_FEA is not None)


if __name__ == "__main__":
    unittest.main()
