"""
Sean Engelstad
: 
Demo analysis of a stiffened panel
NOTE : copy u*iHat+v*jHat+w*kHat for paraview
"""

import ml_buckling as mlb
from mpi4py import MPI
import numpy as np
import pandas as pd
import os

comm = MPI.COMM_WORLD

# I think what happens when h becomes very low for large a,b the norm of K is too small and that affects the soln
# to prevent low thickness problems => just make overall plate size smaller

nom_geometry = mlb.StiffenedPlateGeometry(
    a=0.3, 
    b=0.1,
    h=5e-3,
    num_stiff=1,
    w_b=6e-3,
    t_b=0.0,
    h_w=5e-3,
    t_w=8e-3, # if the wall thickness is too low => stiffener crimping failure happens
)

material = mlb.CompositeMaterial(
    E11=138e9, #Pa
    E22=8.96e9,
    G12=7.1e9,
    nu12=0.30,
    ply_angles=[0,90,0,90],
    ply_fractions=[0.25, 0.25, 0.25, 0.25],
    ref_axis=[1,0,0],
)

nom_panel = mlb.StiffenedPlateAnalysis(
    comm=comm,
    geometry=nom_geometry,
    stiffener_material=material,
    plate_material=material,
)

# initial panel analysis for MAC
nom_panel.pre_analysis(
    global_mesh_size=0.03,
    exx=nom_panel.affine_exx,
    exy=0.0,
    clamped=False,
    edge_pt_min=5,
    edge_pt_max=40,
    _make_rbe=False # would like to change this to True
)

# predict the actual eigenvalue
pred_lambda = nom_panel.predict_crit_load(exx=nom_panel.affine_exx)
_tacs_eigvals, errors = nom_panel.run_buckling_analysis(sigma=10.0, num_eig=20, write_soln=True)
nom_panel.post_analysis()


new_geometry = mlb.StiffenedPlateGeometry.copy(nom_geometry)
new_geometry.h_w = 20e-3

new_panel = mlb.StiffenedPlateAnalysis.copy(nom_panel)
new_panel.geometry = new_geometry


new_panel.pre_analysis(
    global_mesh_size=0.03,
    exx=new_panel.affine_exx,
    exy=0.0,
    clamped=False,
    edge_pt_min=5,
    edge_pt_max=40,
    _make_rbe=False # would like to change this to True
)

# predict the actual eigenvalue
pred_lambda = new_panel.predict_crit_load(exx=new_panel.affine_exx)
_tacs_eigvals, errors = new_panel.run_buckling_analysis(sigma=10.0, num_eig=20, write_soln=True)
new_panel.post_analysis()

eigvals, permutation = mlb.StiffenedPlateAnalysis.mac_permutation(nom_panel, new_panel)
print(f"eigvals = {eigvals}")
print(f"permutation = {permutation}")