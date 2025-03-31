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

class TestAxial_Metal1(unittest.TestCase):

    def setUp(self):
        # Suppress the specific warning from scipy.optimize
        warnings.filterwarnings("ignore", category=UserWarning, message="delta_grad == 0.0.*")
    
    def test_case1(self):

        # ref in data (bad datapoint probably)
        # eig_FEA, eig_CF, ratio FEA/CF
        # 41.745864,8.720298,4.787206
        # rho0 = 0.9, AR = 1.3, delta = 0.18 (somewhat high delta)

        # for some reason it took mode 2 not mode 0
        # when both are clearly global modes.. weird (double check this)
        # that isn't enough to lower the FEA/CF ratio though
        # maybe heuristic is not consistent enough?

        eig_CF, eig_FEA = get_metal_buckling_load(
            comm,
            rho0=0.900,
            gamma=6.6394,
            num_stiff=4,
            sigma_eig=5.0,
            stiff_AR=15.0,
            plate_slenderness=100.0,
            is_axial=False,
        )

        if comm.rank == 0:
            print(f"{eig_CF=} {eig_FEA=}")
            self.assertTrue(eig_FEA is not None)


if __name__ == "__main__":
    unittest.main()
