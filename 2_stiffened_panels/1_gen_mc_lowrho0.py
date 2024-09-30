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

"""
generate second part of the stiffened panel dataset
* nstiff=10 stiffener dataset for rho0 ~ 0.3 and gamma ~ 3.0 (high stiffening regime)
     we also recover some data for different plyAngle values (different xi)
     rho0 from 0.2 to 3.0 (20 pts)
     gamma from 1.0 to 4.0 (10 pts)
     plyAngle from 0.0 to 45.0 (5 pts)
     stiffAR = 5.0 (makes stiffener crippling happen less often)
"""

# argparse
parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument(
    "--clear", default=False, action=argparse.BooleanOptionalAction
)
# type --no-axial to turn it off
parent_parser.add_argument(
    "--axial", default=True, action=argparse.BooleanOptionalAction
)

parent_parser.add_argument("--nelems", type=int, default=2000)

parent_parser.add_argument("--nrho0", type=int, default=20)
parent_parser.add_argument("--rho0Min", type=float, default=0.2)
parent_parser.add_argument("--rho0Max", type=float, default=3.0)

parent_parser.add_argument("--nGamma", type=int, default=10)
parent_parser.add_argument("--gammaMin", type=float, default=1.0)
parent_parser.add_argument("--gammaMax", type=float, default=4.0)

parent_parser.add_argument("--nxi", type=int, default=5)
parent_parser.add_argument("--plyAngleMin", type=float, default=0.0)
parent_parser.add_argument("--plyAngleMax", type=float, default=45.0)


args = parent_parser.parse_args()

prefix = "Nx" if args.axial else "Nxy"
train_csv = f"{prefix}_stiffened.csv"
raw_csv = f"{prefix}_raw_stiffened.csv"
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
def get_buckling_load(
    rho0, 
    gamma, 
    plyAngle, 
    _nstiff, 
    prev_dict=None,
    solve_buckling=True, 
    first=False
):

    stiff_AR = 5.0 #15.0
    plate_SR = 100.0  # 100.0
    b = 1.0
    h = b / plate_SR  # 10 mm
    nstiff = _nstiff if gamma > 0 else 0

    plate_material = mlb.CompositeMaterial.solvay5320(
        ply_angles=[plyAngle], 
        ply_fractions=[1], 
        ref_axis=[1.0, 0.0, 0.0],
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
    MIN_Y = 5
    MIN_Z = 5  # 5
    N = geometry.num_local
    AR_s = geometry.a / geometry.h_w
    nx = np.ceil(np.sqrt(_nelems / (1.0 / AR + (N - 1) / AR_s)))
    den = 1.0 / AR + (N - 1) * 1.0 / AR_s
    ny = max([np.ceil(nx / AR / N), MIN_Y])
    nz = 3

    # print(f"{nx=} {ny=} {nz=}")

    if solve_buckling:
        stiff_analysis.pre_analysis(
            nx_plate=int(nx),  # 90
            ny_plate=int(ny),  # 30
            nz_stiff=int(nz),  # 5
            nx_stiff_mult=2,
            exx=stiff_analysis.affine_exx if args.axial else 0.0,
            exy=stiff_analysis.affine_exy if not(args.axial) else 0.0,
            clamped=False,
            _make_rbe=False,
            _explicit_poisson_exp=True,
        )

    comm.Barrier()

    if solve_buckling:

        tacs_eigvals, errors = stiff_analysis.run_buckling_analysis(
            sigma=5.0, num_eig=100, write_soln=True  # 50, 100
        )
        stiff_analysis.post_analysis()

        global_lambda_star = None
        rho0 = stiff_analysis.affine_aspect_ratio
        gamma = stiff_analysis.gamma
        rho0_star = rho0 / (1.0 + gamma)**0.25
        
        # only do rho0^* < 1.5 the mode tracking since then we are in the regime of (1,1) mode distortion
        # should be less than 1.5 but this is more conservative
        if prev_dict is not None and prev_dict["shape"] == 1: # ensure (1,1) shape before start mode tracking
            if comm.rank == 0:
                # print(f"{prev_dict=}")
                global_lambda_star = stiff_analysis.get_matching_global_mode(
                    prev_dict["nondimX"],
                    prev_dict["phi"],
                    min_similarity=0.7,
                )

                if global_lambda_star is not None:
                    imode = stiff_analysis.min_global_mode_index
                    is_global = stiff_analysis.is_global_mode(
                        imode=stiff_analysis.min_global_mode_index,
                        local_mode_tol=0.75
                    )
                    if not(is_global):
                        global_lambda_star = None
                        print(f"mode {imode} was tracked but it is a local mode..")
            
        else: # just use heuristic
            global_lambda_star = stiff_analysis.get_mac_global_mode(
                axial=args.axial,
                min_similarity=0.75,  # 0.5
                local_mode_tol=0.75, #0.8
            )

        if global_lambda_star is not None:
            if np.abs(errors[stiff_analysis.min_global_mode_index]) > 1e-8:
                global_lambda_star = None

        if comm.rank == 0:
            stiff_analysis.print_mode_classification()
            print(stiff_analysis)

    else:
        global_lambda_star = None

    # predict the actual eigenvalue
    pred_lambda, mode_type = stiff_analysis.predict_crit_load(axial=args.axial)

    # min_eigval = tacs_eigvals[0]
    # rel_err = (pred_lambda - global_lambda_star) / pred_lambda
    if comm.rank == 0 and solve_buckling:
        print(f"{stiff_analysis.intended_Nxx}")

        print(f"Mode type predicted as {mode_type}")
        print(f"\tmy CF min lambda = {pred_lambda}")
        print(f"\tFEA min lambda = {global_lambda_star}")

    comm.Barrier()

    eig_dict = None
    if solve_buckling and comm.rank == 0:
        eig_dict = {
            "nondimX" : stiff_analysis.nondim_X,
            "phi" : stiff_analysis.min_global_eigmode,
            "shape" : stiff_analysis._min_global_mode_shape
        }

    # returns (CF_eig, FEA_eig) as follows:
    return pred_lambda, global_lambda_star, stiff_analysis, eig_dict


# GENERATE DATA
# -------------

if __name__ == "__main__":
    # import argparse
    import matplotlib.pyplot as plt
    from matplotlib import cm

    rho0_vec = np.geomspace(args.rho0Min, args.rho0Max, args.nrho0)
    gamma_vec = np.geomspace(args.gammaMin, args.gammaMax, args.nGamma)
    plyAngle_vec = np.linspace(args.plyAngleMin, args.plyAngleMax, args.nxi)

    ct = 0
    for iply, plyAngle in enumerate(plyAngle_vec):
        for igamma, gamma in enumerate(gamma_vec):
            # rho0 is inner loop so that we can track the 
            eig_dict = None
            for irho0, rho0 in enumerate(rho0_vec[::-1]):
            
                # for nstiff in range(5, 15+1, 2):
                eig_CF, eig_FEA, stiff_analysis, eig_dict = \
                get_buckling_load(
                    rho0=rho0, 
                    gamma=gamma, 
                    plyAngle=plyAngle, 
                    _nstiff=9, # want a large # of stiffeners so that the modes are more global at low rho0
                    prev_dict=eig_dict
                )

                if comm.rank == 0:
                    print(f"{eig_CF=}, {eig_FEA=}")

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
if args.axial:
    os.system("python _combine_stiff_unstiff_data.py --load Nx")
else:
    os.system("python _combine_stiff_unstiff_data.py --load Nxy")
