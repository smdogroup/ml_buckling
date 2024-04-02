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

geometry = mlb.StiffenedPlateGeometry(
    a=0.3, 
    b=0.1,
    h=5e-3,
    num_stiff=1,
    w_b=6e-3,
    t_b=0.0,
    h_w=20e-3,
    t_w=2e-3, # if the wall thickness is too low => stiffener crimping failure happens
)

material = mlb.CompositeMaterial(
    E11=138e9, #Pa
    E22=8.96e9,
    G12=7.1e9,
    nu12=0.30,
    ply_angles=np.deg2rad([0,90,0,90]),
    ply_fractions=np.array([0.25, 0.25, 0.25, 0.25]),
    ref_axis=np.array([1,0,0]),
)

stiff_analysis = mlb.StiffenedPlateAnalysis(
    comm=comm,
    geometry=geometry,
    stiffener_material=material,
    plate_material=material,
    _make_rbe=False #True
)

stiff_analysis.pre_analysis(
    global_mesh_size=0.03,
    exx=stiff_analysis.affine_exx,
    exy=0.0,
    clamped=False,
    edge_pt_min=5,
    edge_pt_max=40,
)

print(f"exx = {stiff_analysis.affine_exx}")

comm.Barrier()

# predict the actual eigenvalue
pred_lambda = stiff_analysis.predict_crit_load(exx=stiff_analysis.affine_exx)
# exit()

avg_stresses = stiff_analysis.run_static_analysis(write_soln=True)
tacs_eigvals, errors = stiff_analysis.run_buckling_analysis(sigma=10.0, num_eig=20, write_soln=True)
stiff_analysis.post_analysis()

print(f"avg stresses = {avg_stresses}")
print(f"tacs eigvals = {tacs_eigvals}")
print(f"errors = {errors}")

min_eigval = tacs_eigvals[0]
rel_err = (pred_lambda - min_eigval) / pred_lambda
print(f"pred min lambda = {pred_lambda}")
print(f"act min lambda = {min_eigval}")
print(f"rel err = {abs(rel_err)}")