"""
made a mistake where D11,D22 of the panel were not correctly adjusted for centroid offset
at high values of gamma.. 
correcting the data for this..
"""

import argparse
import pandas as pd
import ml_buckling as mlb
from mpi4py import MPI
import numpy as np
import scipy.optimize as sopt

comm = MPI.COMM_WORLD

# argparse
parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--load", type=str, default="Nx")

args = parent_parser.parse_args()

assert args.load in ["Nx", "Nxy"]

df = pd.read_csv(f"raw_data/{args.load}_raw_stiffened.csv")

X = df[["rho_0", "xi", "gamma", "zeta", "eig_FEA", "eig_CF"]].to_numpy()

num_trials = X.shape[0]

new_dict = {
    "rho_0": [],
    "xi": [],
    "gamma": [],
    "zeta": [],
    "eig_FEA": [],
    "eig_CF": [],
}

for itrial in range(num_trials):

    prev_rho0 = X[itrial, 0]
    prev_xi = X[itrial, 1]
    prev_gamma = X[itrial, 2]
    prev_eig_FEA = X[itrial, 4]
    prev_eig_CF = X[itrial, 5]

    print(f"{itrial=}/{num_trials}")
    if prev_gamma < 1.0:
        continue

    # get the old dimensions and plate
    # compute the appropriate a,b,h_w,t_w values to achieve rho_0, gamma
    AR = prev_rho0  # since isotropic
    b = 1.0
    a = b * AR
    stiffAR = 1.0
    SR = 20 if args.load == "Nx" else 100
    h = b / SR
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
        E22=138e9,  # 8.96e9
        G12=138e9 / 2.0 / (1 + nu),
        nu12=nu,
        ply_angles=[0, 90, 0, 90],
        ply_fractions=[0.25] * 4,
        ref_axis=[1, 0, 0],
    )

    stiff_material = plate_material

    def gamma_resid(x):
        _geometry = mlb.StiffenedPlateGeometry(
            a=a, b=b, h=h, num_stiff=3, h_w=stiffAR * x, t_w=x
        )
        stiff_analysis = mlb.StiffenedPlateAnalysis(
            comm=comm,
            geometry=_geometry,
            stiffener_material=stiff_material,
            plate_material=plate_material,
        )
        return prev_gamma - stiff_analysis.gamma_no_centroid

    # approximate the h_w,t_w for gamma
    s_p = b / 4  # num_local = num_stiff + 1
    x_guess = np.power(prev_gamma * s_p * h ** 3 / (1 - nu ** 2), 0.25)
    xopt = sopt.fsolve(func=gamma_resid, x0=x_guess)
    # print(f"x = {xopt}")

    t_w = xopt[0]
    h_w = t_w * stiffAR

    geometry = mlb.StiffenedPlateGeometry(a=a, b=b, h=h, num_stiff=3, h_w=h_w, t_w=t_w)
    stiff_analysis = mlb.StiffenedPlateAnalysis(
        comm=comm,
        geometry=geometry,
        stiffener_material=stiff_material,
        plate_material=plate_material,
    )

    # corrected values
    rho0 = stiff_analysis.affine_aspect_ratio
    xi = stiff_analysis.xi_plate
    gamma = stiff_analysis.gamma
    eig_FEA = (
        prev_eig_FEA * stiff_analysis.affine_exx_no_centroid / stiff_analysis.affine_exx
    )
    eig_CF, _ = stiff_analysis.predict_crit_load(
        exx=stiff_analysis.affine_exx if args.load == "Nx" else 0.0,
        exy=stiff_analysis.affine_exy if args.load == "Ny" else 0.0,
    )
    zeta = stiff_analysis.zeta_plate

    # print out the previous and new values
    print(f"{prev_rho0=} {prev_xi=} {prev_gamma=} {prev_eig_FEA=} {prev_eig_CF=}")
    print(f"{rho0=} {xi=} {gamma=} {eig_FEA=} {eig_CF=}")
    # exit()

    # add to new matrix
    new_dict["rho_0"] += [rho0]
    new_dict["xi"] += [xi]
    new_dict["gamma"] += [gamma]
    new_dict["zeta"] += [zeta]
    new_dict["eig_FEA"] += [eig_FEA]
    new_dict["eig_CF"] += [eig_CF]

my_df = pd.DataFrame(new_dict)
my_df.to_csv(f"raw_data/{args.load}_corr.csv")
