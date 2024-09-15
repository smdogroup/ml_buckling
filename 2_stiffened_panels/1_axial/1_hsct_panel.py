import ml_buckling as mlb
from mpi4py import MPI
import numpy as np
import scipy.optimize as sopt

comm = MPI.COMM_WORLD

import argparse
# choose the aspect ratio and gamma values to evaluate on the panel
parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--rho0", type=float, default=1.0)
parent_parser.add_argument("--gamma", type=float, default=1.0)
parent_parser.add_argument("--stiffAR", type=float, default=10.0)
parent_parser.add_argument("--nstiff", type=int, default=3)
parent_parser.add_argument("--sigma", type=float, default=5.0)
parent_parser.add_argument("--SR", type=float, default=100.0)
parent_parser.add_argument("--b", type=float, default=1.0)
parent_parser.add_argument("--nelems", type=int, default=3000)
parent_parser.add_argument("--nx_stiff_mult", type=int, default=3)
parent_parser.add_argument('--static', default=False, action=argparse.BooleanOptionalAction)
parent_parser.add_argument('--buckling', default=True, action=argparse.BooleanOptionalAction)
parent_parser.add_argument('--rbe', default=True, action=argparse.BooleanOptionalAction)
parent_parser.add_argument('--lamCorr', default=False, action=argparse.BooleanOptionalAction)

args = parent_parser.parse_args()

AR = args.rho0 # since isotropic
stiff_AR = args.stiffAR
# b = 0.1
b = args.b
a = b * AR
# h = 5e-3
h = b / args.SR
nu = 0.3
E = 138e9
G = E / 2.0 / (1 + nu)


plate_material = mlb.CompositeMaterial(
    E11=E,  # Pa
    E22=E,
    G12=G,
    nu12=nu,
    ply_angles=[0, 90, 0, 90],
    ply_fractions=[0.25]*4,
    ref_axis=[1, 0, 0],
)

stiff_material = plate_material

# reverse solve the h_w, t_w dimensions of the stiffener
# to produce gamma
def gamma_resid(x):
    _geometry = mlb.StiffenedPlateGeometry(
        a=a, b=b, h=h, num_stiff=args.nstiff, h_w=stiff_AR*x, t_w=x
    )
    stiff_analysis = mlb.StiffenedPlateAnalysis(
        comm=comm,
        geometry=_geometry,
        stiffener_material=stiff_material,
        plate_material=plate_material,
    )
    return args.gamma - stiff_analysis.old_gamma

# approximate the h_w,t_w for gamma
s_p = b / 4 # num_local = num_stiff + 1
x_guess = np.power(args.gamma*s_p*h**3 / (1-nu**2), 0.25)
xopt = sopt.fsolve(func=gamma_resid, x0=x_guess)
# print(f"x = {xopt}")

t_w = xopt[0]
h_w = stiff_AR * t_w

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
    nx_plate=int(nx), #90
    ny_plate=int(ny), #30
    nz_stiff=int(nz), #5
    nx_stiff_mult=args.nx_stiff_mult,
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
    sigma=args.sigma, num_eig=50, write_soln=True
)

if args.static:
    stiff_analysis.run_static_analysis(write_soln=True)

if args.lamCorr:
    avg_stresses = stiff_analysis.run_static_analysis(write_soln=True)
    lam_corr_fact = stiff_analysis.eigenvalue_correction_factor(in_plane_loads=avg_stresses, axial=True)
    # exit()

stiff_analysis.post_analysis()


global_lambda_star = stiff_analysis.min_global_mode_eigenvalue
if args.lamCorr:
    global_lambda_star *= lam_corr_fact

# predict the actual eigenvalue
pred_lambda,mode_type = stiff_analysis.predict_crit_load(axial=True)

if comm.rank == 0:
    stiff_analysis.print_mode_classification()
    print(stiff_analysis)

if global_lambda_star is None:
    rho_0 = args.rho0; gamma = args.gamma
    print(f"{rho_0=}, {gamma=}, {global_lambda_star=}")
    # exit()

if args.lamCorr:
    global_lambda_star *= lam_corr_fact
    if comm.rank == 0: 
        print(f"{avg_stresses=}")
        print(f"{lam_corr_fact=}")

# min_eigval = tacs_eigvals[0]
# rel_err = (pred_lambda - global_lambda_star) / pred_lambda
if comm.rank == 0:
    x_zeta = np.log(1.0+1e3*stiff_analysis.zeta_plate)
    print(f"{x_zeta=}")
    print(f"{stiff_analysis.intended_Nxx}")

    print(f"Mode type predicted as {mode_type}")
    print(f"\tCF min lambda = {pred_lambda}")
    print(f"\tFEA min lambda = {global_lambda_star}")