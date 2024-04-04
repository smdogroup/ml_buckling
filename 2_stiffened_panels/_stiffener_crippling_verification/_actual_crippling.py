"""
Sean Engelstad
: 
Demo analysis of a stiffened panel
NOTE : copy u*iHat+v*jHat+w*kHat for paraview

Note: the linear static analysis reports the in-plane loads not stresses
as these are the working variables in the shell theory (i.e. sx0 for example is Nx)
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
    ply_angles=[0,90,0,90],
    ply_fractions=[0.25]*4,#*4), #[0.4, 0.1, 0.4, 0.1]
    ref_axis=[1,0,0],
)

stiff_analysis = mlb.StiffenedPlateAnalysis(
    comm=comm,
    geometry=geometry,
    stiffener_material=material,
    plate_material=material,
    _make_rbe=True
)

print(f"Darray stiff = {stiff_analysis.Darray_stiff}")
print(f"xi stiff = {stiff_analysis.xi_stiff}")
print(f"gen poisson stiffener = {stiff_analysis.gen_poisson_stiff}")
# exit()

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

# predicted stiffener crippling eigenvalue
Nx_stiff = 3.7e4 # from linear static analysis (need to change average stresses routine to extract by component now since we don't have the stresses, just in-plane loads)
lambda_norm = 0.47 * stiff_analysis.xi_stiff # since stiff_analysis.gen_poisson_stiff approx 0.2
Darray = stiff_analysis.Darray_stiff
D11 = Darray[0]; D12 = Darray[1]; D22 = Darray[2]; D66 = Darray[3]
Nxcrit_stiff = np.pi**2 * np.sqrt(D11 * D22) / geometry.h_w**2 * lambda_norm
pred_eigval = Nxcrit_stiff / Nx_stiff
act_eigval = tacs_eigvals[0]
rel_err = (act_eigval - pred_eigval) / pred_eigval

print(f"prediction of stiffener crippling eigenvalues")
print(f"pred eigval = {pred_eigval}")
print(f"act eigval = {act_eigval}")
print(f"rel err = {rel_err}")