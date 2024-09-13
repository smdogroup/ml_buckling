import ml_buckling as mlb
from mpi4py import MPI
import numpy as np

"""
Verification case based on:

Stiffened panel stability behaviour and performance gains with plate
prismatic sub-stiffening
D. Quinn, A. Murphy, W. McEwan, F. Lemaitre

The specimen A geometry under axial compression with no geometric imperfections
is considered. The initial buckling load of 74.5 kN is the benchmark Nxx,cr to 
compare against with our code. The panel has three stiffeners and is an isotropic
material with Aluminum 2024-T351 alloy.
"""

comm = MPI.COMM_WORLD

import argparse
# choose the aspect ratio and gamma values to evaluate on the panel
parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument('--static', default=False, action=argparse.BooleanOptionalAction)
parent_parser.add_argument('--buckling', default=True, action=argparse.BooleanOptionalAction)

args = parent_parser.parse_args()

# Aluminum 2024-T351 alloy
E = 73.1e9
nu = 0.33

# TODO : double-check that G matches Aluminum tables
# maybe the transverse shear G23, G13 don't copy correctly
plate_material = mlb.CompositeMaterial(
    E11=E,  # Pa
    E22=E, #8.96e9
    G12=E/2.0/(1+nu),
    nu12=nu,
    ply_angles=[0],
    ply_fractions=[1.0],
    ref_axis=[1, 0, 0],
)

stiff_material = plate_material

design_mass = 1.959 # kg
density = 2780 #kg/m^3
volume = design_mass / density
stiff_volume = 3 * 0.028 * 0.0028 * 0.590
panel_volume = volume - stiff_volume
panel_thick = panel_volume / 0.590 / 0.440
# if comm.rank == 0:
#     print(f"{panel_thick=}")
#     exit()

geometry = mlb.StiffenedPlateGeometry(
    a=0.590, #m
    b=0.440,
    h=panel_thick, #computed h = 2.2478e-3 to match overall mass, might be h=2e-3 though?
    s_p=0.167,
    h_w=0.028,
    t_w=0.0028,
)

stiff_analysis = mlb.StiffenedPlateAnalysis(
    comm=comm,
    geometry=geometry,
    stiffener_material=stiff_material,
    plate_material=plate_material,
)


stiff_analysis.pre_analysis(
    nx_plate=30,
    ny_plate=14,
    nz_stiff=3, #5
    nx_stiff_mult=1,
    exx=stiff_analysis.affine_exx,
    exy=0.0,
    clamped=False,
    _make_rbe=False,
    _explicit_poisson_exp=True,
    side_support=True, # instron not supported on the sides
)

comm.Barrier()

if comm.rank == 0:
    print(stiff_analysis)

if args.static:
    stiff_analysis.run_static_analysis(write_soln=True)

if args.buckling:
    tacs_eigvals, errors = stiff_analysis.run_buckling_analysis(
        sigma=1.0, num_eig=50, write_soln=True
    )

    stiff_analysis.post_analysis()

    global_lambda_star = stiff_analysis.min_global_mode_eigenvalue

    # predict the actual eigenvalue
    pred_lambda,mode_type = stiff_analysis.predict_crit_load(exx=stiff_analysis.affine_exx)

    if comm.rank == 0:
        stiff_analysis.print_mode_classification()
        print(stiff_analysis)

    # min_eigval = tacs_eigvals[0]
    # rel_err = (pred_lambda - global_lambda_star) / pred_lambda
    if comm.rank == 0:
        print(f"Mode type predicted as {mode_type}")
        print(f"\tCF min lambda = {pred_lambda}")
        print(f"\tFEA min lambda = {global_lambda_star}")

    if comm.rank == 0:
        mass = geometry.get_mass(density=2780) # density here in kg/m^3
        # mass reported in kg
        print(f"{mass=}")