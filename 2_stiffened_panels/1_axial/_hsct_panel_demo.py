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

# choose the following as both h_w=t_w=this
# 6e-3, 1e-2, 5e-2

geometry = mlb.StiffenedPlateGeometry(
    a=0.1,
    b=0.1,
    h=5e-3,
    num_stiff=3,
    h_w=6e-3,
    t_w=6e-3,  # if the wall thickness is too low => stiffener crimping failure happens
)

plate_material = mlb.CompositeMaterial(
    E11=138e9,  # Pa
    E22=138e9, #8.96e9
    G12=138e9/2.0/(1.3),
    nu12=0.30,
    ply_angles=[0, 90, 0, 90],
    ply_fractions=[0.25]*4,
    ref_axis=[1, 0, 0],
)

stiff_material = plate_material

stiff_analysis = mlb.StiffenedPlateAnalysis(
    comm=comm,
    geometry=geometry,
    stiffener_material=stiff_material,
    plate_material=plate_material,
)

stiff_analysis.pre_analysis(
    nx_plate=30, #90
    ny_plate=30, #30
    nz_stiff=5, #5
    exx=stiff_analysis.affine_exx,
    exy=0.0,
    clamped=False,
    _make_rbe=True,  
)

comm.Barrier()

if comm.rank == 0:
    print(stiff_analysis)

# predict the actual eigenvalue
pred_lambda,mode_type = stiff_analysis.predict_crit_load(exx=stiff_analysis.affine_exx)
pred_lambda2 = stiff_analysis.predict_crit_load_old(exx=stiff_analysis.affine_exx)

if comm.rank == 0:
    print(f"Mode type predicted as {mode_type}")
    print(f"\tmy pred min lambda = {pred_lambda}")
    if geometry.num_stiff == 1:
        print("CF Middlestedt solution")
    else:
        print("Smeared stiffener model")
    print(f"\tref pred min lambda = {pred_lambda2}")

# # exit()
# avg_stresses = stiff_analysis.run_static_analysis(write_soln=True)
# if comm.rank == 0:
#     print(f"avg stresses = {avg_stresses}")
# exit()

tacs_eigvals, errors = stiff_analysis.run_buckling_analysis(
    sigma=5.0, num_eig=50, write_soln=True
)
# stiff_analysis.write_geom()
stiff_analysis.post_analysis()

global_lambda_star = stiff_analysis.min_global_mode_eigenvalue

# predict the actual eigenvalue
pred_lambda,mode_type = stiff_analysis.predict_crit_load(exx=stiff_analysis.affine_exx)
pred_lambda2 = stiff_analysis.predict_crit_load_old(exx=stiff_analysis.affine_exx)

if comm.rank == 0:
    stiff_analysis.print_mode_classification()
    print(stiff_analysis)

# min_eigval = tacs_eigvals[0]
# rel_err = (pred_lambda - global_lambda_star) / pred_lambda
if comm.rank == 0:
    print(f"Mode type predicted as {mode_type}")
    print(f"\tpred min lambda = {pred_lambda}")
    print(f"\tFEA min lambda = {global_lambda_star}")
    # print(f"\trel err = {abs(rel_err)}")
    if geometry.num_stiff == 1:
        print("CF Middlestedt solution")
    else:
        print("Smeared stiffener model")
    print(f"\tref pred min lambda = {pred_lambda2}")
