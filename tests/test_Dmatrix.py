import ml_buckling as mlb
from mpi4py import MPI
import numpy as np
import unittest

comm = MPI.COMM_WORLD

class TestDmatrix(unittest.TestCase):
    def test_case1(self):
        geometry = mlb.StiffenedPlateGeometry(
            a=1.0,
            b=1.0,
            h=1.0,
            num_stiff=0,
            h_w=1e-3,
            t_w=8e-3,  # if the wall thickness is too low => stiffener crimping failure happens
        )

        material = mlb.CompositeMaterial(
            E11=230e9,  # Pa
            E22=6.6e9,
            G12=4.8e9,
            nu12=0.25,
            # ply_angles=[0],
            # ply_fractions=[1.0],
            ply_angles=[0, 90],
            ply_fractions=[0.5]*4,
            ref_axis=[1, 0, 0],
            symmetric=False,
        )

        stiff_analysis = mlb.StiffenedPlateAnalysis(
            comm=comm,
            geometry=geometry,
            stiffener_material=material,
            plate_material=material,
        )

        # compute D array
        Darray = stiff_analysis.Darray_plate
        ref_Darray = np.array([0.9876e10, 0.1377e9, 0.9876e10, 0.4e9])
        print(f"D array = {Darray}")
        print(f"ref D array = {ref_Darray}")
        
        rel_errs = np.abs(Darray - ref_Darray) / np.max(Darray)
        rel_err = np.max(rel_errs)
        print(f"rel errors = {rel_errs}")
        assert rel_err < 0.01
    
    def test_case2(self):
        geometry = mlb.StiffenedPlateGeometry(
            a=1.0,
            b=1.0,
            h=0.15*8,
            num_stiff=0,
            h_w=1e-3,
            t_w=8e-3,  # if the wall thickness is too low => stiffener crimping failure happens
        )

        material = mlb.CompositeMaterial(
            E11=1.5e5,  # MPa
            E22=2e4,
            G12=5e3,
            nu12=0.3,
            # ply_angles=[0],
            # ply_fractions=[1.0],
            ply_angles=[0, 90,0,90],
            ply_fractions=[0.25]*4,
            ref_axis=[1, 0, 0],
            symmetric=True,
        )

        stiff_analysis = mlb.StiffenedPlateAnalysis(
            comm=comm,
            geometry=geometry,
            stiffener_material=material,
            plate_material=material,
        )

        # compute D array
        Darray = stiff_analysis.Darray_plate
        ref_Darray = np.array([1.5941e4, 874.494, 8836.03, 720.0])
        print(f"D array = {Darray}")
        print(f"ref D array = {ref_Darray}")
        
        rel_errs = np.abs(Darray - ref_Darray) / np.max(Darray)
        rel_err = np.max(rel_errs)
        print(f"rel errors = {rel_errs}")
        assert rel_err < 0.01

if __name__=="__main__":
    unittest.main()