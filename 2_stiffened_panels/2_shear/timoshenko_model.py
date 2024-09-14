import ml_buckling as mlb
from mpi4py import MPI
import numpy as np
import scipy.optimize as sopt

comm = MPI.COMM_WORLD

import argparse
# choose the aspect ratio and gamma values to evaluate on the panel
parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--rho0", type=float, default=6.0)
parent_parser.add_argument("--stiffAR", type=float, default=20.0)
parent_parser.add_argument("--nstiff", type=int, default=1)
parent_parser.add_argument("--SR", type=float, default=100.0)
parent_parser.add_argument("--b", type=float, default=1.0)

# change this one to change gamma right now, gamma can only go so high usually with single-sided stiffeners (like gamma < 10, 15)
parent_parser.add_argument("--hw", type=float, default=0.08)

parent_parser.add_argument("--nelems", type=int, default=2000)
parent_parser.add_argument("--sigma", type=float, default=5.0)
args = parent_parser.parse_args()



AR = args.rho0
b = args.b
a = b * AR
h = b / args.SR # 10 mm

h_w = args.hw
t_w = h_w / args.stiffAR

nu = 0.3
E = 138e9
G = E / 2.0 / (1 + nu)


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

geometry = mlb.StiffenedPlateGeometry(
        a=a, b=b, h=h, num_stiff=1, h_w=h_w, t_w=t_w
    )
stiff_analysis = mlb.StiffenedPlateAnalysis(
    comm=comm,
    geometry=geometry,
    stiffener_material=stiff_material,
    plate_material=plate_material,
)

# adjust AR as best we can
act_rho0 = stiff_analysis.affine_aspect_ratio
AR_mult = act_rho0 / AR
AR /= AR_mult
a = b * AR

# make a new plate geometry
geometry = mlb.StiffenedPlateGeometry(
    a=a, b=b, h=h, num_stiff=args.nstiff, h_w=h_w, t_w=t_w
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
MIN_Z = 5 #5
N = geometry.num_local
AR_s = geometry.a / geometry.h_w
#print(f"AR = {AR}, AR_s = {AR_s}")
nx = np.ceil(np.sqrt(_nelems / (1.0/AR + (N-1) / AR_s)))
den = (1.0/AR + (N-1) * 1.0 / AR_s)
ny = max([np.ceil(nx / AR / N), MIN_Y])
# nz = max([np.ceil(nx / AR_s), MIN_Z])
nz = 3

# my_list = [np.ceil(nx / AR / N), MIN_Y]
# print(f"{my_list=}")

print(f"{nx=} {ny=} {nz=}")

stiff_analysis.pre_analysis(
    nx_plate=int(nx), #90
    ny_plate=int(ny), #30
    nz_stiff=int(nz), #5
    nx_stiff_mult=2,
    # exx=stiff_analysis.affine_exx,
    exx=0.0,
    exy=stiff_analysis.affine_exy,
    clamped=False,
    # _make_rbe=args.rbe, 
    _make_rbe=True,
    _explicit_poisson_exp=False, 
)

# print(f"{}")

comm.Barrier()

if comm.rank == 0:
    print(stiff_analysis)
# exit()

tacs_eigvals, errors = stiff_analysis.run_buckling_analysis(
    sigma=args.sigma, 
    num_eig=50, #50, 100
    write_soln=True
)

# if args.static:
# stiff_analysis.run_static_analysis(write_soln=True)
stiff_analysis.post_analysis()


global_lambda_star = stiff_analysis.min_global_mode_eigenvalue

# predict the actual eigenvalue
pred_lambda,mode_type = stiff_analysis.predict_crit_load(exy=stiff_analysis.affine_exy)

if comm.rank == 0:
    stiff_analysis.print_mode_classification()
    print(stiff_analysis)

# min_eigval = tacs_eigvals[0]
# rel_err = (pred_lambda - global_lambda_star) / pred_lambda
if comm.rank == 0:
    print(f"{stiff_analysis.intended_Nxx}")

    # only for isotropic
    # # timoshenko isotropic closed-form
    # beta = geometry.a / geometry.b
    # gamma = stiff_analysis.gamma / 2.0
    # delta = stiff_analysis.delta / 2.0
    # timosh_crit = 1e10
    # for m in range(1,100):
    #     # in book we assume m = 1
    #     temp = ( (1.0 + beta**2/m**2)**2 + 2.0 * gamma ) * m**2 / beta**2 / (1.0 + 2.0 * delta)
    #     if temp < timosh_crit:
    #         timosh_crit = temp

    print(f"Mode type predicted as {mode_type}")
    # print(f"\ttimoshenko CF lambda = {timosh_crit}")
    print(f"\tmy CF min lambda = {pred_lambda}")
    print(f"\tFEA min lambda = {global_lambda_star}")