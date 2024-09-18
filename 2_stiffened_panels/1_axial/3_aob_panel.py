import ml_buckling as mlb
from mpi4py import MPI
import numpy as np
import scipy.optimize as sopt

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

a = 0.7023
b = 1.882
h = 0.0135 # m
h_w = 0.0767 #m
t_w = 0.01524
# test if stiffAR higher => you do get higher xi and rho0 global
# t_w = h_w / 20.0
s_p = 0.3871

nu = 0.3
E = 138e9
G = E / 2.0 / (1 + nu)

plate_material = mlb.CompositeMaterial(
    E11=E,  # Pa
    E22=E,
    G12=G,
    nu12=nu,
    ply_angles=[0, 90, 0, 90],
    ply_fractions=[0.25] * 4,
    ref_axis=[1, 0, 0],
)

stiff_material = plate_material

geometry = mlb.StiffenedPlateGeometry(
    a=a, b=b, h=h, s_p=s_p, h_w=h_w, t_w=t_w
)
stiff_analysis = mlb.StiffenedPlateAnalysis(
    comm=comm,
    geometry=geometry,
    stiffener_material=stiff_material,
    plate_material=plate_material,
)

_nelems = 2000
AR = a / b
# MIN_Y = 20 / geometry.num_local
MIN_Y = 5
MIN_Z = 5  # 5
N = geometry.num_local
AR_s = geometry.a / geometry.h_w
# print(f"AR = {AR}, AR_s = {AR_s}")
nx = np.ceil(np.sqrt(_nelems / (1.0 / AR + (N - 1) / AR_s)))
den = 1.0 / AR + (N - 1) * 1.0 / AR_s
# print(f"{AR=}")
# print(f"{N=}")
# print(f"{den=}")
# print(f"{nx=}")
ny = max([np.ceil(nx / AR / N), MIN_Y])
# nz = max([np.ceil(nx / AR_s), MIN_Z])
nz = 5

# my_list = [np.ceil(nx / AR / N), MIN_Y]
# print(f"{my_list=}")

print(f"{nx=} {ny=} {nz=}")

stiff_analysis.pre_analysis(
    nx_plate=int(nx),  # 90
    ny_plate=int(ny),  # 30
    nz_stiff=int(nz),  # 5
    nx_stiff_mult=3,
    exx=stiff_analysis.affine_exx,
    exy=0.0,
    clamped=False,
    # _make_rbe=args.rbe,
    _make_rbe=False,
    _explicit_poisson_exp=True,
)

# print(f"{}")

comm.Barrier()

if comm.rank == 0:
    print(stiff_analysis)
# exit()

tacs_eigvals, errors = stiff_analysis.run_buckling_analysis(
    sigma=10.0, num_eig=100, write_soln=True
)

if args.static:
    stiff_analysis.run_static_analysis(write_soln=True)

stiff_analysis.post_analysis()


# global_lambda_star = stiff_analysis.min_global_mode_eigenvalue
global_lambda_star = stiff_analysis.get_mac_global_mode(
    axial=True,
    min_similarity=0.7,
    local_mode_tol=0.7,
)

# predict the actual eigenvalue
pred_lambda, mode_type = stiff_analysis.predict_crit_load(axial=True)

if comm.rank == 0:
    stiff_analysis.print_mode_classification()
    print(stiff_analysis)
    # exit()

# min_eigval = tacs_eigvals[0]
# rel_err = (pred_lambda - global_lambda_star) / pred_lambda
if comm.rank == 0:
    print(f"Mode type predicted as {mode_type}")
    print(f"\tCF min lambda = {pred_lambda}")
    print(f"\tFEA min lambda = {global_lambda_star}")
