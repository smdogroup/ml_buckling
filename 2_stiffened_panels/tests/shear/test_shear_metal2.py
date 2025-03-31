import numpy as np
import ml_buckling as mlb
import pandas as pd
from mpi4py import MPI
import os, argparse, sys
import unittest
sys.path.append("../")

from _utils import get_metal_buckling_load
import warnings

comm = MPI.COMM_WORLD

class TestShear_Metal1(unittest.TestCase):

    def setUp(self):
        # Suppress the specific warning from scipy.optimize
        warnings.filterwarnings("ignore", category=UserWarning, message="delta_grad == 0.0.*")
    
    def test_case1(self):

        # ref in data (higher gamma, but barely higher than prev lower gamma)
        # maybe needs more stiffeners?

        # was getting N11^cr bar = 30.723, now getting 42.816

        eig_CF, eig_FEA = get_metal_buckling_load(
            comm,
            rho0=0.623,
            gamma=9.114,
            num_stiff=3, # 3 worked
            sigma_eig=5.0,
            stiff_AR=15.0,
            plate_slenderness=100.0,
            is_axial=False,
            ant_plot_debug=True,
        )

        ratio = eig_FEA / eig_CF

        if comm.rank == 0:
            print(f"{eig_CF=} {eig_FEA=} {ratio=}")
            self.assertTrue(eig_FEA is not None)


if __name__ == "__main__":
    unittest.main()
