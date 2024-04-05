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

geometry = mlb.StiffenedPlateGeometry(
    a=0.3, 
    b=0.1,
    h=5e-3,
    num_stiff=1,
    w_b=6e-3,
    t_b=0.0,
    h_w=15e-3,
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

stiff_analysis = mlb.StiffenedPlateAnalysis(
    comm=comm,
    geometry=geometry,
    stiffener_material=material,
    plate_material=material,
)


tacs_eigvals = []
CF_eigvals = []
rel_errors = []
stiff_AR = 5.0

data_folder = os.path.join(os.getcwd(), "data")
if not os.path.exists(data_folder):
    os.mkdir(data_folder)

log10_hw = np.linspace(-3, -1, 50)
hw_vec = np.power(10, log10_hw)
for i,h_w in enumerate(hw_vec):
    b_w = h_w / stiff_AR
    stiff_analysis.geometry.h_w = h_w
    stiff_analysis.geometry.b_w = b_w

    stiff_analysis.pre_analysis(
        global_mesh_size=0.03,
        exx=stiff_analysis.affine_exx,
        exy=0.0,
        clamped=False,
        edge_pt_min=5,
        edge_pt_max=40,
        _make_rbe=False
    )

    # predict the actual eigenvalue
    pred_lambda = stiff_analysis.predict_crit_load(exx=stiff_analysis.affine_exx)
    _tacs_eigvals, errors = stiff_analysis.run_buckling_analysis(sigma=10.0, num_eig=20, write_soln=False)
    stiff_analysis.post_analysis()

    tacs_eigvals += [np.real(_tacs_eigvals[0])]
    CF_eigvals += [pred_lambda]
    rel_err = (pred_lambda - np.real(_tacs_eigvals[0])) / pred_lambda
    rel_errors += [rel_err]


    df_dict = {
        "h_w" : [h_w],
        "tacs_eig" : [tacs_eigvals[-1]],
        "CF_eig" : [CF_eigvals[-1]],
        "rel_err" : [rel_errors[-1]]
    }
    df = pd.DataFrame(df_dict)
    df.to_csv(os.path.join(data_folder, "1_hstudy.csv"), mode='a' if i != 0 else 'w', header = i == 0)

    stiff_analysis.post_analysis()