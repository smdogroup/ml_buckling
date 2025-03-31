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

        # orig inputs (other than known ones above) to recreate above
        # num_stiff=6, ply_angle=28, SR=140

        # need more stiffeners to converge buckling load 
        # my N_s convergence criterion must not have worked well enough
        # N_s=6  : Nbar_12^cr = 147
        # N_s=7  : Nbar_12^cr = 179
        # N_s=8  : Nbar_12^cr = 199
        # N_s=9  : Nbar_12^cr = 209.66
        # N_s=10 : Nbar_12^cr = 218.624 (is this converging?)
        # N_s=11 : Nbar_12^cr = 223.944
        # N_s=12 : Nbar_12^cr = 233.521 (delta, xi, zeta keep changing but so is )
        # N_s=13 : Nbar_12^cr = 224.075

        # log buckling load should be about 0.42 higher at least so that log(N12bar^cr) : 4.996 => 5.42

        # need stiffeners to converge the FEA/CF ratio
        # N_s=6  : 2.154
        # N_s=7  : 2.455
        # N_s=8  : 2.7478
        # N_s=9  : 2.9086
        # N_s=10 : 3.0529
        # N_s=11 : 3.2049 (xi is now 0.65, zeta 9e-5)
        # N_s=12 : 3.1585
        # N_s=13 : 3.2424 (gamma is now 6.020, gamma has decreased)
        # N_s=14 : 3.16108 (gamma is now 5.8318)

        # consensus solution : need stricter N_s convergence tolerance
        # it was nowhere near N_s converged, tricky to find N_s convergence when R(xhat)
        # solver eventually fails though at high N_s..

        # these inputs are pretty close.. why much lower than curve?
        eig_CF, eig_FEA = get_composite_buckling_load(
            comm,
            rho0=0.314144,
            gamma=6.500360,
            num_stiff=14,
            sigma_eig=5.0,
            stiff_AR=15.0,
            ply_angle=28.0,
            composite_material=mlb.CompositeMaterial.hexcelIM7,
            plate_slenderness=200.0,
            is_axial=False,
            ant_plot_debug=False, # turn on to see ant contour plot
        )

        ratio = eig_FEA / eig_CF

        if comm.rank == 0:
            print(f"{eig_CF=} {eig_FEA=} {ratio=}")
            self.assertTrue(eig_FEA is not None)


if __name__ == "__main__":
    unittest.main()

