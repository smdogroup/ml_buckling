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

args = parent_parser.parse_args()

AR = args.rho0 # since isotropic
b = 0.1
a = b * AR
h = 5e-3
nu = 0.3

plate_material = mlb.CompositeMaterial(
    E11=138e9,  # Pa
    E22=138e9, #8.96e9
    G12=138e9/2.0/(1+nu),
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
        a=a, b=b, h=h, num_stiff=3, h_w=x, t_w=x
    )
    stiff_analysis = mlb.StiffenedPlateAnalysis(
        comm=comm,
        geometry=_geometry,
        stiffener_material=stiff_material,
        plate_material=plate_material,
    )
    return args.gamma - stiff_analysis.gamma

# approximate the h_w,t_w for gamma
s_p = b / 4 # num_local = num_stiff + 1
x_guess = np.power(args.gamma*s_p*h**3 / (1-nu**2), 0.25)
xopt = sopt.fsolve(func=gamma_resid, x0=x_guess)
print(f"x = {xopt}")

h_w = t_w = xopt[0]

geometry = mlb.StiffenedPlateGeometry(
    a=a, b=b, h=h, num_stiff=3, h_w=h_w, t_w=t_w
)
stiff_analysis = mlb.StiffenedPlateAnalysis(
    comm=comm,
    geometry=geometry,
    stiffener_material=stiff_material,
    plate_material=plate_material,
)

_nelems = 3000
MIN_Y = 20 / geometry.num_local
MIN_Z = 10 #5
N = geometry.num_local
AR_s = geometry.a / geometry.h_w
#print(f"AR = {AR}, AR_s = {AR_s}")
nx = np.ceil(np.sqrt(_nelems / (1.0/AR + (N-1) / AR_s)))
ny = max(np.ceil(nx / AR / N), MIN_Y)
nz = max(np.ceil(nx / AR_s), MIN_Z)

stiff_analysis.pre_analysis(
    nx_plate=int(nx), #90
    ny_plate=int(ny), #30
    nz_stiff=int(nz), #5
    exx=stiff_analysis.affine_exx,
    exy=0.0,
    clamped=False,
    _make_rbe=True,  
)

comm.Barrier()

if comm.rank == 0:
    print(stiff_analysis)
    # print(f"gamma = {gamma}")
# exit()

# predict the actual eigenvalue
pred_lambda,mode_type = stiff_analysis.predict_crit_load(exx=stiff_analysis.affine_exx)
pred_lambda2 = stiff_analysis.predict_crit_load_old(exx=stiff_analysis.affine_exx)

if comm.rank == 0:
    print(f"Mode type predicted as {mode_type}")
    print(f"\tmy pred min lambda = {pred_lambda}")
    if geometry.num_stiff == 1:
        print("CF Middlestedt solution")
    else:
        print("Smeared stiffener model")
    print(f"\tref pred min lambda = {pred_lambda2}")

# # exit()
# avg_stresses = stiff_analysis.run_static_analysis(write_soln=True)
# if comm.rank == 0:
#     print(f"avg stresses = {avg_stresses}")
# exit()

sigma = 5.0
# for i in range(5):
tacs_eigvals, errors = stiff_analysis.run_buckling_analysis(
    sigma=sigma, num_eig=50, write_soln=False
)
    # # stiff_analysis.write_geom()
    # stiff_analysis.post_analysis()
    # if stiff_analysis.min_global_mode_eigenvalue is not None:
    #     break
    # else:
    #     sigma *= 5.0 # keep increasing sigma if it failed

stiff_analysis.post_analysis()


global_lambda_star = stiff_analysis.min_global_mode_eigenvalue

# predict the actual eigenvalue
pred_lambda,mode_type = stiff_analysis.predict_crit_load(exx=stiff_analysis.affine_exx)
pred_lambda2 = stiff_analysis.predict_crit_load_old(exx=stiff_analysis.affine_exx)

if comm.rank == 0:
    stiff_analysis.print_mode_classification()
    print(stiff_analysis)

if global_lambda_star is None:
    rho_0 = args.rho0; gamma = args.gamma
    print(f"{rho_0=}, {gamma=}, {global_lambda_star=}")
    exit()

# min_eigval = tacs_eigvals[0]
# rel_err = (pred_lambda - global_lambda_star) / pred_lambda
if comm.rank == 0:
    print(f"Mode type predicted as {mode_type}")
    print(f"\tpred min lambda = {pred_lambda}")
    print(f"\tFEA min lambda = {global_lambda_star}")
    # print(f"\trel err = {abs(rel_err)}")
    if geometry.num_stiff == 1:
        print("CF Middlestedt solution")
    else:
        print("Smeared stiffener model")
    print(f"\tref pred min lambda = {pred_lambda2}")
