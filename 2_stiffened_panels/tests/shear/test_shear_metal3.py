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

class TestShear_Metal3(unittest.TestCase):

    def setUp(self):
        # Suppress the specific warning from scipy.optimize
        warnings.filterwarnings("ignore", category=UserWarning, message="delta_grad == 0.0.*")
    
    def test_case1(self):

        eig_FEA_list = []

        # these inputs are pretty close.. why much lower than curve?
        for Ns in range(1, 10+1):
        # my_Ns = 3
        # for Ns in range(my_Ns, my_Ns+1):
            eig_CF, eig_FEA = get_metal_buckling_load(
                comm,
                rho0=0.3,
                gamma=3.04,
                num_stiff=Ns,
                sigma_eig=10.0,
                stiff_AR=15.0,
                plate_slenderness=100,
                is_axial=False,
                ant_plot_debug=False, # turn on to see ant contour plot
            )
            eig_FEA_list += [eig_FEA]


        Ns_list = [_ for _ in range(1, 10+1)]
        if comm.rank == 0:
            print(f"{Ns_list=}")
            print(f"{eig_FEA_list}")
 

if __name__ == "__main__":
    unittest.main()

