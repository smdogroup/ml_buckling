import ml_buckling as mlb
from mpi4py import MPI
import numpy as np
import scipy.optimize as sopt

comm = MPI.COMM_WORLD

import argparse

# choose the aspect ratio and gamma values to evaluate on the panel
parent_parser = argparse.ArgumentParser(add_help=False)
# recommended by Stephen E. that higher # of stiffeners than 1
# and stiffAR = 20 before is too high, should be lower.
parent_parser.add_argument("--stiffAR", type=float, default=5.0)
parent_parser.add_argument("--nstiff", type=int, default=6)
parent_parser.add_argument("--SR", type=float, default=100.0)
parent_parser.add_argument("--b", type=float, default=1.0)
parent_parser.add_argument("--plyAngle", type=float, default=0.0)
parent_parser.add_argument(
    "--orthotropic", default=False, action=argparse.BooleanOptionalAction
)
parent_parser.add_argument(
    "--axial", default=False, action=argparse.BooleanOptionalAction
)


# change this one to change gamma right now, gamma can only go so high usually with single-sided stiffeners (like gamma < 10, 15)
parent_parser.add_argument("--rho0", type=float, default=0.3)
parent_parser.add_argument("--gamma", type=float, default=2.0)

# MAC settings
parent_parser.add_argument("--minSim", type=float, default=0.7)
parent_parser.add_argument("--globLocal", type=float, default=0.7)

parent_parser.add_argument("--nelems", type=int, default=2000)
parent_parser.add_argument("--sigma", type=float, default=5.0)
args = parent_parser.parse_args()


AR = args.rho0
b = args.b
a = b * AR
h = b / args.SR  # 10 mm

if args.orthotropic:
    plate_material = mlb.CompositeMaterial.solvay5320(
        ply_angles=[45, -30.0, 0.0], 
        ply_fractions=[0.3, 0.3, 0.4], 
        ref_axis=[1.0, 0.0, 0.0],
    )
else:

    nu = 0.3
    E = 138e9
    G = E / 2.0 / (1 + nu)

    # doesn't actually matter what value of E since disp control
    # so can be titanium, aluminum, etc.
    plate_material = mlb.CompositeMaterial(
        E11=E,  # Pa
        E22=E,
        G12=G,
        nu12=nu,
        ply_angles=[0],
        ply_fractions=[1.0],
        ref_axis=[1, 0, 0],
    )

stiff_material = plate_material


def gamma_rho0_resid(x):
    h_w = x[0]
    AR = x[1]
    t_w = h_w / args.stiffAR

    a = AR * b

    geometry = mlb.StiffenedPlateGeometry(
        a=a, b=b, h=h, num_stiff=args.nstiff, h_w=h_w, t_w=t_w
    )
    stiff_analysis = mlb.StiffenedPlateAnalysis(
        comm=comm,
        geometry=geometry,
        stiffener_material=stiff_material,
        plate_material=plate_material,
    )

    return [
        args.rho0 - stiff_analysis.affine_aspect_ratio,
        args.gamma - stiff_analysis.gamma,
    ]


xopt = sopt.fsolve(func=gamma_rho0_resid, x0=(0.08 * args.gamma / 11.25, args.rho0))

h_w = xopt[0]
AR = xopt[1]
a = b * AR

# make a new plate geometry
geometry = mlb.StiffenedPlateGeometry(
    a=a, b=b, h=h, num_stiff=args.nstiff, h_w=h_w, t_w=h_w / args.stiffAR
)
stiff_analysis = mlb.StiffenedPlateAnalysis(
    comm=comm,
    geometry=geometry,
    stiffener_material=stiff_material,
    plate_material=plate_material,
)

_nelems = args.nelems
# MIN_Y = 20 / geometry.num_local
MIN_Y = 5
MIN_Z = 5  # 5
N = geometry.num_local
AR_s = geometry.a / geometry.h_w
# print(f"AR = {AR}, AR_s = {AR_s}")
nx = np.ceil(np.sqrt(_nelems / (1.0 / AR + (N - 1) / AR_s)))
den = 1.0 / AR + (N - 1) * 1.0 / AR_s
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
    nx_stiff_mult=2,
    exx=stiff_analysis.affine_exx if args.axial else 0.0,
    exy=stiff_analysis.affine_exy if not(args.axial) else 0.0,
    clamped=False,
    # _make_rbe=args.rbe,
    _make_rbe=False,
    _explicit_poisson_exp=True,
)

# print(f"{}")

comm.Barrier()

tacs_eigvals, errors = stiff_analysis.run_buckling_analysis(
    sigma=args.sigma, num_eig=100, write_soln=True  # 50, 100
)

# if args.static:
# stiff_analysis.run_static_analysis(write_soln=True)
stiff_analysis.post_analysis()


# global_lambda_star = stiff_analysis.min_global_mode_eigenvalue
global_lambda_star = stiff_analysis.get_mac_global_mode(
    axial=args.axial,
    min_similarity=args.minSim,
    local_mode_tol=args.globLocal,
)

# predict the actual eigenvalue
pred_lambda, mode_type = stiff_analysis.predict_crit_load(axial=True)

if comm.rank == 0:
    stiff_analysis.print_mode_classification()
    print(stiff_analysis)

# also get SS slopes
if comm.rank == 0:
    imode = stiff_analysis._min_global_imode
    for xedge in [True, False]:
        stiff_analysis.get_nondim_slopes(
            imode, xedge=xedge, m=1, n=1
        )

# min_eigval = tacs_eigvals[0]
# rel_err = (pred_lambda - global_lambda_star) / pred_lambda
if comm.rank == 0:
    # timoshenko isotropic closed-form
    beta = geometry.a / geometry.b
    gamma = stiff_analysis.gamma / 2.0
    delta = stiff_analysis.delta / 2.0
    timosh_crit = 1e10
    for m in range(1, 100):
        # in book we assume m = 1
        temp = (
            ((1.0 + beta ** 2 / m ** 2) ** 2 + 2.0 * gamma)
            * m ** 2
            / beta ** 2
            / (1.0 + 2.0 * delta)
        )
        if temp < timosh_crit:
            timosh_crit = temp

    print(f"Mode type predicted as {mode_type}")
    print(f"\ttimoshenko CF lambda = {timosh_crit}")
    print(f"\tmy CF min lambda = {pred_lambda}")
    print(f"\tFEA min lambda = {global_lambda_star}")
