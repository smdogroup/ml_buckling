"""
Sean Engelstad
August 2024, GT SMDO Lab
Goal is to generate a dataset for pure uniaxial buckling of stiffened panels.

NOTE : copy u*iHat+v*jHat+w*kHat for paraview
"""

import ml_buckling as mlb
import pandas as pd
import numpy as np, os, argparse
from mpi4py import MPI
import scipy.optimize as sopt

comm = MPI.COMM_WORLD

# argparse
parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument('--clear', default=False, action=argparse.BooleanOptionalAction)
parent_parser.add_argument('--lamCorr', default=False, action=argparse.BooleanOptionalAction)
parent_parser.add_argument("--nrho0", type=int, default=20)
parent_parser.add_argument("--nGamma", type=int, default=20)
parent_parser.add_argument("--nelems", type=int, default=3000)
parent_parser.add_argument("--rho0Min", type=float, default=0.3)
parent_parser.add_argument("--gammaMin", type=float, default=0.01)
parent_parser.add_argument("--rho0Max", type=float, default=5.0)
parent_parser.add_argument("--gammaMax", type=float, default=500.0)

args = parent_parser.parse_args()

train_csv = "Nx_stiffened.csv"
raw_csv = "Nx_raw_stiffened.csv"
cpath = os.path.dirname(__file__)
data_folder = os.path.join(cpath, "raw_data")
if not os.path.exists(data_folder) and comm.rank == 0:
    os.mkdir(data_folder)

train_csv_path = os.path.join(data_folder, train_csv)
raw_csv_path = os.path.join(data_folder, raw_csv)

if args.clear and comm.rank == 0:
    if os.path.exists(train_csv_path):
        os.remove(train_csv_path)
    if os.path.exists(raw_csv_path):
        os.remove(raw_csv_path)
comm.Barrier()

# DEFINE ANALYSIS ROUTINE
# -----------------------
def get_buckling_load(rho_0, gamma):
    # previously ply_angle was input => but we found that the intended strain state
    # is not achieved in stiffened panel case when not isotropic..
    # so only trust isotropic stiffened panel models for now (least deviation in near-unstiffened case)

    # compute the appropriate a,b,h_w,t_w values to achieve rho_0, gamma
    AR = rho_0 # since isotropic
    b = 0.1
    a = b * AR
    stiffAR = 1.0
    h = 5e-3
    nu = 0.3

    # found that the intended strain state is very hard to achieve with the stiffened panel models
    # only the isotropic case actually achieves the intended strain state in these datasets.
    # namely highly orthotropic panels in their linear static analyses are meant to only have ex0 (only mid-plane axial strain nonzero)
    #  however, they end up with ey0 positive and tensile, ex1 and ey1 bending curvatures. Then the buckling loads don't compare well 
    #  to closed-form since the intended strain state is way different than the closed-form solution and the unstiffened panel models => 
    #  which results in a poor ML model. 
    #  Only the isotropic models actually achieve the intended strain state right now for stiffened panels and are usable for ML.
    #  future work could fix the FEA models potentially to achieve the intended strain states for all stiffened panel materials.
    # for this reason material variation is only considered in the unstiffened panel dataset
    # and we assume the same xi_slope from the unstiffened dataset when we extrapolate this data.
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

    def gamma_resid(x):
        _geometry = mlb.StiffenedPlateGeometry(
            a=a, b=b, h=h, num_stiff=3, h_w=stiffAR*x, t_w=x
        )
        stiff_analysis = mlb.StiffenedPlateAnalysis(
            comm=comm,
            geometry=_geometry,
            stiffener_material=stiff_material,
            plate_material=plate_material,
        )
        return gamma - stiff_analysis.gamma

    # approximate the h_w,t_w for gamma
    s_p = b / 4 # num_local = num_stiff + 1
    x_guess = np.power(gamma*s_p*h**3 / (1-nu**2), 0.25)
    xopt = sopt.fsolve(func=gamma_resid, x0=x_guess)
    # print(f"x = {xopt}")

    t_w = xopt[0]
    h_w = t_w * stiffAR

    geometry = mlb.StiffenedPlateGeometry(
        a=a, b=b, h=h, num_stiff=3, h_w=h_w, t_w=t_w
    )
    stiff_analysis = mlb.StiffenedPlateAnalysis(
        comm=comm,
        geometry=geometry,
        stiffener_material=stiff_material,
        plate_material=plate_material,
    )
    
    _nelems = args.nelems
    MIN_Y = 20 / geometry.num_local # 20
    MIN_Z = 5 #5
    N = geometry.num_local
    AR_s = geometry.a / geometry.h_w
    #print(f"AR = {AR}, AR_s = {AR_s}")
    nx = np.ceil(np.sqrt(_nelems / (1.0/AR + (N-1) / AR_s)))
    nx = min(nx, 50)
    ny = max(np.ceil(nx / AR / N), MIN_Y)
    nz = max(np.ceil(nx / AR_s), MIN_Z)

    stiff_analysis.pre_analysis(
        nx_plate=int(nx), #90
        ny_plate=int(ny), #30
        nz_stiff=int(nz), #5
        nx_stiff_mult=3,
        exx=stiff_analysis.affine_exx,
        exy=0.0,
        clamped=False,
        _make_rbe=True,  
    )

    comm.Barrier()

    if comm.rank == 0:
        print(stiff_analysis)

    # guess using closed-form
    # pred_lambda,_ = stiff_analysis.predict_crit_load(exx=stiff_analysis.affine_exx)
    # sigma = pred_lambda
    # sigma = 10.0 if pred_lambda > 10.0 else 5.0
    sigma = 5.0

    tacs_eigvals, errors = stiff_analysis.run_buckling_analysis(
        sigma=sigma, num_eig=50, write_soln=False
    )

    if args.lamCorr:
        avg_stresses = stiff_analysis.run_static_analysis(write_soln=True)
        lam_corr_fact = stiff_analysis.eigenvalue_correction_factor(in_plane_loads=avg_stresses, axial=True)
        # exit()

    stiff_analysis.post_analysis()


    global_lambda_star = stiff_analysis.min_global_mode_eigenvalue
    if args.lamCorr:
        global_lambda_star *= lam_corr_fact

    if abs(errors[0]) > 1e-4:
        global_lambda_star = None

    # predict the actual eigenvalue
    pred_lambda,mode_type = stiff_analysis.predict_crit_load(exx=stiff_analysis.affine_exx)

    if comm.rank == 0:
        stiff_analysis.print_mode_classification()
        print(stiff_analysis)

    if global_lambda_star is None:
        print(f"{rho_0=}, {gamma=}, {global_lambda_star=}")
        # exit()

    if args.lamCorr:
        global_lambda_star *= lam_corr_fact
        if comm.rank == 0: 
            print(f"{avg_stresses=}")
            print(f"{lam_corr_fact=}")
        # exit()

    # min_eigval = tacs_eigvals[0]
    # rel_err = (pred_lambda - global_lambda_star) / pred_lambda
    if comm.rank == 0:
        print(f"Mode type predicted as {mode_type}")
        print(f"\tCF min lambda = {pred_lambda}")
        print(f"\tFEA min lambda = {global_lambda_star}")
        print("--------------------------------------\n", flush=True)

    # returns (CF_eig, FEA_eig) as follows:
    return pred_lambda, global_lambda_star, stiff_analysis


# GENERATE DATA
# -------------

if __name__=="__main__":
    # import argparse
    import matplotlib.pyplot as plt
    from matplotlib import cm

    rho0_vec = np.geomspace(args.rho0Min, args.rho0Max, args.nrho0)
    gamma_vec = np.geomspace(args.gammaMin, args.gammaMax, args.nGamma)

    ct = 0
    for igamma,gamma in enumerate(gamma_vec):
        for irho0,rho0 in enumerate(rho0_vec):
            eig_CF, eig_FEA, stiff_analysis = get_buckling_load(
                rho_0=rho0, 
                gamma=gamma
            )
            if comm.rank == 0:
                print(f"{eig_CF=}, {eig_FEA=}")
            if eig_FEA is None: 
                eig_FEA = np.nan # just leave value as almost zero..
                continue

            # write out as you go so you can see the progress and if run gets killed you don't lose it all
            if comm.rank == 0:
                ct += 1
                raw_data_dict = {
                    # training parameter section
                    "rho_0": [stiff_analysis.affine_aspect_ratio],
                    "xi": [stiff_analysis.xi_plate],
                    "gamma": [stiff_analysis.gamma],
                    "zeta": [stiff_analysis.zeta_plate],
                    "eig_FEA": [np.real(eig_FEA)],
                    "eig_CF": [eig_CF],
                }
                raw_df = pd.DataFrame(raw_data_dict)
                first_write = ct == 1 and args.clear
                raw_df.to_csv(raw_csv_path, mode="w" if first_write else "a", header=first_write)


    