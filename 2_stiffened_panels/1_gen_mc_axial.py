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
parent_parser.add_argument(
    "--clear", default=False, action=argparse.BooleanOptionalAction
)
parent_parser.add_argument("--nrho0", type=int, default=50)
parent_parser.add_argument("--nGamma", type=int, default=20)
parent_parser.add_argument("--nelems", type=int, default=2000)
parent_parser.add_argument("--rho0Min", type=float, default=0.3)
parent_parser.add_argument("--gammaMin", type=float, default=0.05)
parent_parser.add_argument("--rho0Max", type=float, default=15.0)
parent_parser.add_argument("--gammaMax", type=float, default=15.0)

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
def get_buckling_load(rho0, gamma, solve_buckling=True, first=False):

    stiff_AR = 20.0
    plate_SR = 100.0  # 100.0
    b = 1.0
    h = b / plate_SR  # 10 mm
    nu = 0.3
    E = 138e9
    G = E / 2.0 / (1 + nu)
    # h_w = 0.08 #0.08
    # t_w = h_w / stiff_AR # 0.005
    nstiff = 1 if gamma > 0 else 0

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
        t_w = h_w / stiff_AR

        a = AR * b

        geometry = mlb.StiffenedPlateGeometry(
            a=a, b=b, h=h, num_stiff=nstiff, h_w=h_w, t_w=t_w
        )
        stiff_analysis = mlb.StiffenedPlateAnalysis(
            comm=comm,
            geometry=geometry,
            stiffener_material=stiff_material,
            plate_material=plate_material,
        )

        return [rho0 - stiff_analysis.affine_aspect_ratio, gamma - stiff_analysis.gamma]

    xopt = sopt.fsolve(func=gamma_rho0_resid, x0=(0.08 * gamma / 11.25, rho0))

    h_w = xopt[0]
    AR = xopt[1]
    a = b * AR

    # make a new plate geometry
    geometry = mlb.StiffenedPlateGeometry(
        a=a, b=b, h=h, num_stiff=nstiff, h_w=h_w, t_w=h_w / stiff_AR
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
    nz = 3

    # my_list = [np.ceil(nx / AR / N), MIN_Y]
    # print(f"{my_list=}")

    # print(f"{nx=} {ny=} {nz=}")

    if solve_buckling:
        stiff_analysis.pre_analysis(
            nx_plate=int(nx),  # 90
            ny_plate=int(ny),  # 30
            nz_stiff=int(nz),  # 5
            nx_stiff_mult=2,
            exx=stiff_analysis.affine_exx,
            exy=0.0,
            clamped=False,
            # _make_rbe=args.rbe,
            _make_rbe=False,
            _explicit_poisson_exp=True,
        )

    comm.Barrier()

    # if comm.rank == 0 and solve_buckling:
    #     print(stiff_analysis)
    # exit()

    if solve_buckling:

        tacs_eigvals, errors = stiff_analysis.run_buckling_analysis(
            sigma=5.0, num_eig=50, write_soln=True  # 50, 100
        )

        # if args.static:
        # stiff_analysis.run_static_analysis(write_soln=True)
        stiff_analysis.post_analysis()

        global_lambda_star = stiff_analysis.get_mac_global_mode(
            axial=True,
            min_similarity=0.7,  # 0.5
            local_mode_tol=0.7,
        )
        # global_lambda_star = stiff_analysis.min_global_mode_eigenvalue

        # if global_lambda_star is None:
        #     global_lambda_star = np.nan

        if comm.rank == 0:
            stiff_analysis.print_mode_classification()
            print(stiff_analysis)

    else:
        global_lambda_star = None

    # predict the actual eigenvalue
    pred_lambda, mode_type = stiff_analysis.predict_crit_load(axial=True)

    # min_eigval = tacs_eigvals[0]
    # rel_err = (pred_lambda - global_lambda_star) / pred_lambda
    if comm.rank == 0 and solve_buckling:
        print(f"{stiff_analysis.intended_Nxx}")

        print(f"Mode type predicted as {mode_type}")
        print(f"\tmy CF min lambda = {pred_lambda}")
        print(f"\tFEA min lambda = {global_lambda_star}")

    comm.Barrier()

    # exit()
    # return [global_lambda_star, pred_lambda, timosh_crit]

    # returns (CF_eig, FEA_eig) as follows:
    return pred_lambda, global_lambda_star, stiff_analysis


# GENERATE DATA
# -------------

if __name__ == "__main__":
    # import argparse
    import matplotlib.pyplot as plt
    from matplotlib import cm

    rho0_vec = np.geomspace(args.rho0Min, args.rho0Max, args.nrho0)
    gamma_vec = np.geomspace(args.gammaMin, args.gammaMax, args.nGamma)

    ct = 0
    for igamma, gamma in enumerate(gamma_vec):
        for irho0, rho0 in enumerate(rho0_vec):
            eig_CF, eig_FEA, stiff_analysis = get_buckling_load(rho0=rho0, gamma=gamma)
            if comm.rank == 0:
                print(f"{eig_CF=}, {eig_FEA=}")

            # for first 2 irho0 if eig_FEA doesn't exist
            # add CF values in its place to help the ML model training in this regime
            # # will result in conservative training in this low rho0 regime
            # if irho0 < 2 and gamma >= 1.0 and eig_FEA is None:
            #     eig_FEA = eig_CF

            if eig_FEA is None:
                eig_FEA = np.nan  # just leave value as almost zero..
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
                raw_df.to_csv(
                    raw_csv_path, mode="w" if first_write else "a", header=first_write
                )


# then combine the stiff and unstiff data
# os.system("python _combine_stiff_unstiff_data.py --load Nx")
