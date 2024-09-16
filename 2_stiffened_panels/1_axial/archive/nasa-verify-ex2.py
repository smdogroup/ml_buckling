import ml_buckling as mlb
from mpi4py import MPI
import numpy as np

"""
Verification case based on:

NASA TR 2215

Buckling Loads of Stiffened
Panels Subjected to Combined
Longitudinal Compression and
Shear:ResultsObtained
With PASCO, EAL, and STAGS
Computer Programs

NOTE : ignore the mode classification which says stiffener crippling..
it should say inconclusive or mixed here..

Example 2
page 11 - overall panel dimensions
page 15 - mesh
page 27 - Metal blade-stiffened panel with thin skin

The minimum eigenvalues for this very thin stiffAR is based on local mode though
    and I care about verifying global modes.
"""

comm = MPI.COMM_WORLD

import argparse

# choose the aspect ratio and gamma values to evaluate on the panel
parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument(
    "--static", default=False, action=argparse.BooleanOptionalAction
)
parent_parser.add_argument(
    "--buckling", default=True, action=argparse.BooleanOptionalAction
)

args = parent_parser.parse_args()

E = 72.4e9
nu = 0.32

plate_material = mlb.CompositeMaterial(
    E11=E,  # Pa
    E22=E,  # 8.96e9
    G12=E / 2.0 / (1 + nu),
    nu12=nu,
    ply_angles=[0, 90] * 2,
    ply_fractions=[0.25] * 4,
    ref_axis=[1, 0, 0],
)

stiff_material = plate_material


geometry = mlb.StiffenedPlateGeometry(
    a=0.762,
    b=0.762,
    h=0.00213,
    num_stiff=6,  # 7 panel sections
    h_w=0.03434,
    t_w=0.00147,
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
    nz_stiff=3,  # 5
    nx_stiff_mult=1,
    exx=stiff_analysis.affine_exx,
    # exx = 1e-3,
    exy=0.0,
    clamped=False,
    _make_rbe=False,
    _explicit_poisson_exp=True,
)

comm.Barrier()

if comm.rank == 0:
    print(stiff_analysis)
# exit()

if args.static:
    stiff_analysis.run_static_analysis(write_soln=True)

if args.buckling:
    tacs_eigvals, errors = stiff_analysis.run_buckling_analysis(
        sigma=5.0, num_eig=50, write_soln=True
    )

    stiff_analysis.post_analysis()

    global_lambda_star = stiff_analysis.min_global_mode_eigenvalue

    # predict the actual eigenvalue
    pred_lambda, mode_type = stiff_analysis.predict_crit_load(
        exx=stiff_analysis.affine_exx
    )

    if comm.rank == 0:
        stiff_analysis.print_mode_classification()
        print(stiff_analysis)

    # min_eigval = tacs_eigvals[0]
    # rel_err = (pred_lambda - global_lambda_star) / pred_lambda
    if comm.rank == 0:
        print(f"Mode type predicted as {mode_type}")
        print(f"\tCF min lambda = {pred_lambda}")
        print(f"\tFEA min lambda = {global_lambda_star}")
        x_zeta = np.log(1.0 + 1e3 * stiff_analysis.zeta_plate)
        print(f"{x_zeta=}")
        print(f"Nx = {stiff_analysis.intended_Nxx}")
