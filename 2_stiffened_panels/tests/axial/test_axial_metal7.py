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

class TestAxial_Metal4(unittest.TestCase):

    def setUp(self):
        # Suppress the specific warning from scipy.optimize
        warnings.filterwarnings("ignore", category=UserWarning, message="delta_grad == 0.0.*")
    
    def test_from_script(self):

        # lessons learned from this case (not running in main 1_gen_data.py)
        # beta = 30 then it fails, beta = 60 (slenderness) it runs, needs to be light enough

        # case from finite element dataset
        eig_CF, eig_FEA = get_metal_buckling_load(
            comm,
            rho0=0.900, # also 0.378
            gamma=6.6394,
            num_stiff=8, # suddenly 1,2 fail and 3 works, need to loop until not None sometimes 
            sigma_eig=10.0,
            stiff_AR=15.0,
            b=1.0,
            ant_hs_ind=0, # 0 default
            # plate_slenderness=27.38,
            plate_slenderness=100.0,
            is_axial=True,
        )

        if comm.rank == 0:
            print(f"{eig_CF=} {eig_FEA=}")
            self.assertTrue(eig_FEA is not None)

if __name__ == "__main__":
    unittest.main()
