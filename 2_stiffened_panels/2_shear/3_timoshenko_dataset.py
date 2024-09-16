import ml_buckling as mlb
from mpi4py import MPI
import numpy as np
import scipy.optimize as sopt
import niceplots
import pandas as pd

"""
As of right now => just have to manually remove 
distorted shear modes at the boundary of when it first starts 
picking up from None to not None FEA buckling loads
"""

comm = MPI.COMM_WORLD

import argparse

# choose the aspect ratio and gamma values to evaluate on the panel
parent_parser = argparse.ArgumentParser(add_help=False)
args = parent_parser.parse_args()


def shear_load(rho0, gamma, solve_buckling=True, first=False):

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

    _nelems = 2000
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
            # exx=stiff_analysis.affine_exx,
            exx=0.0,
            exy=stiff_analysis.affine_exy,
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
            sigma=5.0, num_eig=100, write_soln=True  # 50, 100
        )

        # if args.static:
        # stiff_analysis.run_static_analysis(write_soln=True)
        stiff_analysis.post_analysis()

        global_lambda_star = stiff_analysis.get_mac_global_mode(
            axial=False,
            # min_similarity=0.7, #0.5
            local_mode_tol=0.7,
        )
        # global_lambda_star = stiff_analysis.min_global_mode_eigenvalue

        if global_lambda_star is None:
            global_lambda_star = np.nan

        if comm.rank == 0:
            stiff_analysis.print_mode_classification()
            print(stiff_analysis)

    else:
        global_lambda_star = None

    # predict the actual eigenvalue
    pred_lambda, mode_type = stiff_analysis.predict_crit_load(axial=False)

    # # timoshenko isotropic closed-form
    # beta = geometry.a / geometry.b
    # TS_gamma = stiff_analysis.gamma / 2.0
    # TS_delta = stiff_analysis.delta / 2.0
    # timosh_crit = 1e10
    # for m in range(1,100):
    #     # in book we assume m = 1
    #     temp = ( (1.0 + beta**2/m**2)**2 + 2.0 * TS_gamma ) * m**2 / beta**2 / (1.0 + 2.0 * TS_delta)
    #     if temp < timosh_crit:
    #         timosh_crit = temp

    # min_eigval = tacs_eigvals[0]
    # rel_err = (pred_lambda - global_lambda_star) / pred_lambda
    if comm.rank == 0 and solve_buckling:
        print(f"{stiff_analysis.intended_Nxx}")

        print(f"Mode type predicted as {mode_type}")
        # print(f"\ttimoshenko CF lambda = {timosh_crit}")
        print(f"\tmy CF min lambda = {pred_lambda}")
        print(f"\tFEA min lambda = {global_lambda_star}")

        # write the data to csv  after each buckling solve
        my_df_dict = {
            "rho0": [rho0],
            "gamma": [gamma],
            "eig_CF": [pred_lambda],
            "eig_FEA": [global_lambda_star],
        }
        my_df = pd.DataFrame(my_df_dict)
        my_df.to_csv("shearCF-FEA.csv", header=first, mode="w" if first else "a")

    comm.Barrier()

    # exit()

    return [global_lambda_star, pred_lambda, None]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rho0_min = 0.2
    rho0_max = 20.0
    n_FEA = 50  # 50 (should be 50 in order to not plot as densely..)

    # TODO : make it so we can set gamma here also and adaptively select hw

    gamma_vec = [0.0, 2.0, 5.0, 10.0]
    # gamma_vec = [0.0]

    plt.style.use(niceplots.get_style())

    plt.figure("this")
    colors = mlb.six_colors2[:4][::-1]

    for igamma, gamma in enumerate(gamma_vec[::-1]):

        n_CF = n_FEA
        rho0_CF = np.geomspace(rho0_min, rho0_max, n_CF)
        N11_CF = np.array(
            [shear_load(rho0, gamma=gamma, solve_buckling=False)[1] for rho0 in rho0_CF]
        )

        if comm.rank == 0:
            # label="closed-form"
            plt.plot(rho0_CF, N11_CF, "-", label=None, color=colors[igamma])
            # plt.show()

        # rho0_CFT = np.geomspace(rho0_min, rho0_max, 100)
        # N11_CFT = np.array([
        #     axial_load(rho0, gamma=1.0, solve_buckling=False)[2] for rho0 in rho0_CFT
        # ])

        # if comm.rank == 0:
        #     plt.plot(rho0_CFT, N11_CFT, "-", label="timosh")

        if comm.rank == 0:
            print("done with closed-form timoshenko")
            # exit()

        rho0_FEA = np.geomspace(rho0_min, rho0_max, n_FEA)
        N11_FEA = np.array(
            [
                shear_load(rho0, gamma=gamma, first=igamma == 0 and irho0 == 0)[0]
                for irho0, rho0 in enumerate(rho0_FEA)
            ]
        )

        if comm.rank == 0:
            # FEA
            plt.plot(
                rho0_FEA,
                N11_FEA,
                "o",
                label=r"$\gamma = " + f"{gamma:.1f}" + r"$",
                color=colors[igamma],
            )

        comm.Barrier()

    # finish making the plot
    if comm.rank == 0:
        plt.legend()
        # plt.title(r"$\gamma = 11.25$")
        plt.xlabel(r"$\rho_0$")
        plt.ylabel(r"$N_{12,cr}^*$")
        plt.xscale("log")
        plt.yscale("log")
        plt.margins(x=0.05, y=0.05)
        # plt.show()
        plt.savefig("shear-CF-vs-FEA-demo.png", dpi=400)
