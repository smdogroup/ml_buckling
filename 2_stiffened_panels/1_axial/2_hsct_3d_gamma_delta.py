"""
Sean Engelstad
: 
Demo analysis of a stiffened panel
NOTE : copy u*iHat+v*jHat+w*kHat for paraview
"""

import ml_buckling as mlb
from mpi4py import MPI
import numpy as np
import scipy.optimize as sopt
import pandas as pd


import argparse

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--npts", type=int, default=10)
parent_parser.add_argument("--nelems", type=int, default=3000)
parent_parser.add_argument("--rho0Min", type=float, default=0.3)
parent_parser.add_argument("--gammaMin", type=float, default=0.01)
parent_parser.add_argument("--rho0Max", type=float, default=5.0)
parent_parser.add_argument("--gammaMax", type=float, default=100.0)
parent_parser.add_argument('--debug', default=False, action=argparse.BooleanOptionalAction)
parent_parser.add_argument('--lamCorr', default=False, action=argparse.BooleanOptionalAction)

args = parent_parser.parse_args()

if args.debug:
    args.nelems = 3000

comm = MPI.COMM_WORLD

def get_buckling_load(rho_0, gamma):

    # compute the appropriate a,b,h_w,t_w values to achieve rho_0, gamma
    AR = rho_0 # since isotropic
    b = 0.1
    a = b * AR
    stiffAR = 1.0
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
    print(f"x = {xopt}")

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
        nx_stiff_mult=3,
        exx=stiff_analysis.affine_exx,
        exy=0.0,
        clamped=False,
        _make_rbe=True,  
    )

    comm.Barrier()

    if comm.rank == 0:
        print(stiff_analysis)


    tacs_eigvals, errors = stiff_analysis.run_buckling_analysis(
        sigma=5.0, num_eig=50, write_soln=False
    )

    if args.lamCorr:
        avg_stresses = stiff_analysis.run_static_analysis(write_soln=True)
        lam_corr_fact = stiff_analysis.eigenvalue_correction_factor(in_plane_loads=avg_stresses, axial=True)
        # exit()

    stiff_analysis.post_analysis()


    global_lambda_star = stiff_analysis.min_global_mode_eigenvalue
    if args.lamCorr:
        global_lambda_star *= lam_corr_fact

    # predict the actual eigenvalue
    pred_lambda,mode_type = stiff_analysis.predict_crit_load(exx=stiff_analysis.affine_exx)

    if comm.rank == 0:
        stiff_analysis.print_mode_classification()
        print(stiff_analysis)

    if global_lambda_star is None:
        rho_0 = args.rho0; gamma = args.gamma
        print(f"{rho_0=}, {gamma=}, {global_lambda_star=}")
        exit()

    if args.lamCorr:
        global_lambda_star *= lam_corr_fact
        if comm.rank == 0: 
            print(f"{avg_stresses=}")
            print(f"{lam_corr_fact=}")

    # min_eigval = tacs_eigvals[0]
    # rel_err = (pred_lambda - global_lambda_star) / pred_lambda
    if comm.rank == 0:
        print(f"Mode type predicted as {mode_type}")
        print(f"\tCF min lambda = {pred_lambda}")
        print(f"\tFEA min lambda = {global_lambda_star}")
        print("--------------------------------------\n", flush=True)

    # returns (CF_eig, FEA_eig) as follows:
    return pred_lambda, global_lambda_star

if __name__=="__main__":
    # import argparse
    import matplotlib.pyplot as plt
    from matplotlib import cm

    # now make side-by-side contour plots comparing the stiffened 
    # panel buckling loads btw CF and FEA

    if args.debug:
        # args.npts = 2; 
        args.rho0Min = 0.5; args.rho0Max = 2.0
        args.gammaMin = 0.01; args.gammaMax = 100.0

    n =  args.npts
    # rho0_vec = np.geomspace(0.1, 10.0, n)
    rho0_vec = np.geomspace(args.rho0Min, args.rho0Max, n)
    gamma_vec = np.geomspace(args.gammaMin, args.gammaMax, n)
    # can't get quite up to gamma = 1000.0 because then there are only local / stiffener modes
    # gamma_vec = np.geomspace(0.01, 100.0, n)
    RHO0, GAMMA = np.meshgrid(rho0_vec, gamma_vec)
    print(f"{RHO0=}")
    print(f"{RHO0[0,:]}")
    print(f"{rho0_vec=}")
    print(f"{gamma_vec=}")
    # exit()

    CF = np.zeros((n,n)); FEA = np.zeros((n,n))
    for igamma,gamma in enumerate(gamma_vec):
        for irho0,rho0 in enumerate(rho0_vec):
            eig_CF, eig_FEA = get_buckling_load(rho_0=rho0, gamma=gamma)
            if comm.rank == 0:
                print(f"{eig_CF=}, {eig_FEA=}")
            if eig_FEA is None: eig_FEA = 1e-14 # just leave value as almost zero..
            CF[igamma,irho0] = eig_CF
            FEA[igamma,irho0] = eig_FEA

            # write out as you go so you can see the progress and if run gets killed you don't lose it all
            if comm.rank == 0:
                df_dict = {
                    "log(rho0)" : np.array([np.log(rho0)]),
                    "log(1+gamma)" : np.array([np.log(1.0+gamma)]),
                    "log(N11-CF)" : np.array([np.log(eig_CF)]),
                    "log(N11-FEA)" : np.array([np.log(eig_FEA)]),
                }
                df = pd.DataFrame(df_dict)
                first_write = igamma == 0 and irho0 == 0
                df.to_csv("2-hsct-axial.csv", mode="w" if first_write else "a", header=first_write)

    LOG_RHO0 = np.log(RHO0)
    LOG_GAMMA = np.log(1.0 + GAMMA)
    LOG_CF = np.log(CF)
    LOG_FEA = np.log(FEA)

    # # write all of the data to csv files by flattening the matrices
    # if comm.rank == 0:
    #     df_dict = {
    #         "log(rho0)" : LOG_RHO0.flatten(order='c'),
    #         "log(1+gamma)" : LOG_GAMMA.flatten(order='c'),
    #         "log(N11-CF)" : LOG_CF.flatten(order='c'),
    #         "log(N11-FEA)" : LOG_FEA.flatten(order='c'),
    #     }
    #     df = pd.DataFrame(df_dict)
    #     df.to_csv("2-hsct-axial.csv")

    # TODO : write the data out to npy or csv files..

    if comm.rank == 0:
        print("Now plotting the results in 3d..")

        # fig, axs = plt.subplots(2)
        # fig.suptitle('Vertically stacked subplots')
        # axs[0].contourf(LOG_RHO0, LOG_GAMMA, LOG_CF)
        # axs[1].contourf(LOG_RHO0, LOG_GAMMA, LOG_FEA)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # Plot the surface.
        surf1 = ax.plot_surface(LOG_RHO0, LOG_GAMMA, LOG_CF, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        mynorm = surf1.norm
        surf2 = ax.plot_surface(LOG_RHO0, LOG_GAMMA, LOG_FEA, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        surf2.set_norm(mynorm)

        fig.colorbar(surf1, shrink=0.5, aspect=5)

        plt.show()

        # also plot the difference in eigenvalues and make a colorplot
        fig2, ax2 = plt.subplots(layout='constrained')
        cs = ax2.contourf(LOG_RHO0, LOG_GAMMA, LOG_FEA - LOG_CF)
        # plt.colorbar()
        fig2.colorbar(cs, ax=ax2, shrink=0.9)
        plt.show()


    
    