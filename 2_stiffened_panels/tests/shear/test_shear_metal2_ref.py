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

class TestShear_Metal2Ref(unittest.TestCase):

    def setUp(self):
        # Suppress the specific warning from scipy.optimize
        warnings.filterwarnings("ignore", category=UserWarning, message="delta_grad == 0.0.*")
    
    def test_case1(self):

        # ref in data (higher gamma, but barely higher than prev lower gamma)
        # maybe needs more stiffeners?

        # ,rho_0,xi,gamma,log10(zeta),num_stiff,material,eig_FEA,eig_CF,time
        # ,1.111003,0.845665,9.119695,-3.893144,2,hexcelIM7,30.728554,11.124209,19.999994

        # with metal and xi = 0.769 I get eig_FEA = 22.647 and eig_CF = 3.0340

        eig_CF, eig_FEA = get_metal_buckling_load(
            comm,
            rho0=1.111,
            gamma=9.119695,
            num_stiff=2, # 3 worked
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