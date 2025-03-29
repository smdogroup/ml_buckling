"""
Sean Engelstad, Feb 2024
GT SMDO Lab
"""
import ml_buckling as mlb
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD

plate_material = mlb.CompositeMaterial.solvay5320(
    ply_angles=[30.0],
    ply_fractions=[1.0],
    ref_axis=[1.0, 0.0, 0.0],
)

geometry = mlb.StiffenedPlateGeometry(
    a=5.0,
    b=1.0,
    h=0.005,
    num_stiff=0,
    h_w=0.01, t_w=0.0,
)
stiff_analysis = mlb.StiffenedPlateAnalysis(
    comm=comm,
    geometry=geometry,
    stiffener_material=plate_material,
    plate_material=plate_material,
)

stiff_analysis.pre_analysis(
    nx_plate=31,  # 90
    ny_plate=31,  # 30
    nz_stiff=2,  # 5
    nx_stiff_mult=2,
    exx=stiff_analysis.affine_exx,
    exy=0.0,
    clamped=False,
    _make_rbe=False,
    _explicit_poisson_exp=True, # for stiffener only
)
stiff_analysis.run_static_analysis(base_path=None, write_soln=True)

tacs_eigvals, errors = stiff_analysis.run_buckling_analysis(
    sigma=10.0, num_eig=100, write_soln=True  # 50, 100
)
# whereas here we get lam1 = 2.56 with 1e-14 error in the eigenvalue solver
# seems like the error was bad on that solution? Need a way from python to check error in solution..

print(f"tacs eigvals = {tacs_eigvals}")
print(f"errors = {errors}")
