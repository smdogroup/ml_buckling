import numpy as np
import ml_buckling as mlb
import pandas as pd
from mpi4py import MPI
import os, argparse, sys
import unittest
from _utils import get_metal_buckling_load, axial_load_old
import warnings

comm = MPI.COMM_WORLD

class TestAxial_Metal2(unittest.TestCase):

    def setUp(self):
        # Suppress the specific warning from scipy.optimize
        warnings.filterwarnings("ignore", category=UserWarning, message="delta_grad == 0.0.*")
    
    def test_new(self):

        # lessons learned from this case (comparing to _axial/7_stiff_dataset.py gamma = 3 code)
            # essential : 
            #    stiffAR is 15 not 20 (needs to be low enough)
            # non-essential params:
            #    sigma_eig can be 5 or 10, no effect
            #    nstiff can be 5, 9 not 1
            # was having problems with stiffAR = 20.0 before because it stiffener cripples

        eig_CF, eig_FEA = get_metal_buckling_load(
            comm,
            rho0=0.6, # less than 0.6 fails..
            gamma=3.0,
            num_stiff=1, #5
            sigma_eig=5.0,
            stiff_AR=15.0,
            # sigma_eig=10.0,
            # stiff_AR=15.0,
            b=1.0,
            ant_hs_ind=0, # 0 default
            plate_slenderness=100.0,
            is_axial=True,
        )

        if comm.rank == 0:
            print(f"{eig_CF=} {eig_FEA=}")
            self.assertTrue(eig_FEA is not None)

if __name__ == "__main__":
    unittest.main()
