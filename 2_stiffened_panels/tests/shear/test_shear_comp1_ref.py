import numpy as np
import ml_buckling as mlb
import pandas as pd
from mpi4py import MPI
import os, argparse, sys
import unittest
sys.path.append("../")

from _utils import get_composite_buckling_load
import warnings

comm = MPI.COMM_WORLD

class TestShear_Metal3Ref(unittest.TestCase):

    def setUp(self):
        # Suppress the specific warning from scipy.optimize
        warnings.filterwarnings("ignore", category=UserWarning, message="delta_grad == 0.0.*")
    
    def test_case1(self):

        # ref in data (higher gamma, but barely higher than prev lower gamma)
        # maybe needs more stiffeners?

        # ,rho_0,xi,gamma,log10(zeta),num_stiff,material,eig_FEA,eig_CF,time
        # ,0.314144,0.876552,6.500360,-4.038407,6,hexcelIM7,147.871176,72.767961,62.480811

        # these inputs are pretty close.. why much lower than curve?
        eig_CF, eig_FEA = get_composite_buckling_load(
            comm,
            rho0=0.314144,
            gamma=6.500360,
            num_stiff=6,
            sigma_eig=5.0,
            stiff_AR=15.0,
            ply_angle=28.0,
            composite_material=mlb.CompositeMaterial.hexcelIM7,
            plate_slenderness=140.0,
            is_axial=False,
            ant_plot_debug=False, # turn on to see ant contour plot
        )

        ratio = eig_FEA / eig_CF

        if comm.rank == 0:
            print(f"{eig_CF=} {eig_FEA=} {ratio=}")
            self.assertTrue(eig_FEA is not None)


if __name__ == "__main__":
    unittest.main()

