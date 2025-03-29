import numpy as np
import ml_buckling as mlb
import pandas as pd
from mpi4py import MPI
import os, argparse, sys
import unittest
from _utils import get_metal_buckling_load, axial_load_old
import warnings

comm = MPI.COMM_WORLD

class TestAxial_Metal3_Ref(unittest.TestCase):

    def setUp(self):
        # Suppress the specific warning from scipy.optimize
        warnings.filterwarnings("ignore", category=UserWarning, message="delta_grad == 0.0.*")
    
    def test_from_script(self):

        # lessons learned from this case (not running in main 1_gen_data.py)
        #  num_stiff matter?
        #  sigma_eig matter?
        #  plate_slenderness matter? (was more slender)

        # case from finite element dataset
        eig_CF, eig_FEA = get_metal_buckling_load(
            comm,
            rho0=0.8, # also 0.378
            gamma=1.1,
            num_stiff=9, #5
            sigma_eig=5.0,
            stiff_AR=15.0,
            b=1.0,
            ant_hs_ind=0, # 0 default
            plate_slenderness=27.38,
            is_axial=True,
        )

        if comm.rank == 0:
            print(f"{eig_CF=} {eig_FEA=}")
            self.assertTrue(eig_FEA is not None)

if __name__ == "__main__":
    unittest.main()
