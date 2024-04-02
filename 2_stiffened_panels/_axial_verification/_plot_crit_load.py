"""
Sean Engelstad
: 
Demo analysis of a stiffened panel
NOTE : copy u*iHat+v*jHat+w*kHat for paraview
"""

import ml_buckling as mlb
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD

# I think what happens when h becomes very low for large a,b the norm of K is too small and that affects the soln
# to prevent low thickness problems => just make overall plate size smaller

geometry = mlb.StiffenedPlateGeometry(
    a=0.3, 
    b=0.1,
    h=1e-3,
    num_stiff=1,
    w_b=6e-3,
    t_b=0.0,
    h_w=50e-3,
    t_w=1e-3, # if the wall thickness is too low => stiffener crimping failure happens
)

material = mlb.CompositeMaterial(
    E11=138e9, #Pa
    E22=8.96e9,
    G12=7.1e9,
    nu12=0.30,
    ply_angles=np.deg2rad([0,90,0,90]),
    ply_fractions=np.array([0.25]*4), #np.array([0.4, 0.1, 0.4, 0.1])
    ref_axis=np.array([1,0,0]),
)

stiff_analysis = mlb.StiffenedPlateAnalysis(
    comm=comm,
    geometry=geometry,
    stiffener_material=material,
    plate_material=material,
    _make_rbe=True #True
)

# predict the actual eigenvalue
hw_vec = np.linspace(1e-4, 20e-3, 100)
lambda_vec = np.zeros((100,))
for ih,h_w in enumerate(hw_vec):
    geometry.h_w = h_w
    lambda_vec[ih] = stiff_analysis.predict_crit_load(exx=stiff_analysis.affine_exx, mode_loop=True)
    #print(f"pred lambda = {lambda_vec}")

plt.plot(hw_vec*1000, lambda_vec)
plt.xlabel("stiffener height (mm)")
# plt.xscale('log')
plt.ylabel(r"$\lambda$")
plt.savefig("_case1.png")