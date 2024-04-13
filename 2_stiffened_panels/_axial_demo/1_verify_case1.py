"""
Sean Engelstad
: 
Demo analysis of a stiffened panel
NOTE : copy u*iHat+v*jHat+w*kHat for paraview
"""

import ml_buckling as mlb
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

# I think what happens when h becomes very low for large a,b the norm of K is too small and that affects the soln
# to prevent low thickness problems => just make overall plate size smaller

case = 1
assert case in [1,2]
if case == 1:
    h_w = 10e-3
elif case == 2:
    h_w = 1e-3
# very short stiffener with bad element AR leads to lower eigenvalue (easier to buckle?)
# think this is an artifact of the FEA solution.
elif case == 3: 
    h_w = 1e-4

geometry = mlb.StiffenedPlateGeometry(
    a=0.3,
    b=0.1,
    h=5e-3,
    num_stiff=1,
    h_w=h_w,
    t_w=8e-3,  # if the wall thickness is too low => stiffener crimping failure happens
)

material = mlb.CompositeMaterial(
    E11=138e9,  # Pa
    E22=8.96e9,
    G12=7.1e9,
    nu12=0.30,
    ply_angles=[0, 90, 0, 90],
    ply_fractions=[0.25, 0.25, 0.25, 0.25],
    ref_axis=[1, 0, 0],
)

stiff_analysis = mlb.StiffenedPlateAnalysis(
    comm=comm,
    geometry=geometry,
    stiffener_material=material,
    plate_material=material,
)

stiff_analysis.pre_analysis(
    nx_plate=30,
    ny_plate=30,
    nz_stiff=20,
    exx=stiff_analysis.affine_exx,
    exy=0.0,
    clamped=False,
    _make_rbe=True,  # True
)

comm.Barrier()

# avg_stresses = stiff_analysis.run_static_analysis(write_soln=True)
tacs_eigvals, errors = stiff_analysis.run_buckling_analysis(
    sigma=10.0, num_eig=50, write_soln=True
)
stiff_analysis.post_analysis()

global_lambda_star = stiff_analysis.min_global_mode_eigenvalue

# predict the actual eigenvalue
pred_lambda,mode_type = stiff_analysis.predict_crit_load(exx=stiff_analysis.affine_exx)

if comm.rank == 0:
    print(stiff_analysis)
    stiff_analysis.print_mode_classification()

min_eigval = tacs_eigvals[0]
rel_err = (pred_lambda - global_lambda_star) / pred_lambda
if comm.rank == 0:
    print(f"Mode type predicted as {mode_type}")
    print(f"\tpred min lambda = {pred_lambda}")
    print(f"\tFEA min lambda = {global_lambda_star}")
    print(f"\trel err = {abs(rel_err)}")
