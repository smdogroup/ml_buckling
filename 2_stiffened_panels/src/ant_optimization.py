import ml_buckling as mlb    
import numpy as np
from scipy.optimize import minimize
from opt_utils import gamma_rho_resid_wrapper

import warnings

# Suppress the specific warning about delta_grad
warnings.filterwarnings("ignore", message="delta_grad == 0.0.*")

def ant_optimization(
    comm,
    plate_material:mlb.CompositeMaterial,
    stiff_material:mlb.CompositeMaterial,
    nstiff:int,
    stiff_AR:float,
    b:float,
    h:float,
    rho0:float,
    gamma:float,
    debug:bool=False,
    plot_debug:bool=False,
    hs_ind:int=0,
):
    
    print(f"ant opt: {comm.rank} checkpt0", flush=True)
     
    target_log_gamma = np.log10(1.0+gamma)
    # target_log_rho0 = np.log10(rho0)

    # define residual equations
    # -------------------------

    gamma_rho0_resid = gamma_rho_resid_wrapper(
        comm, plate_material, stiff_material, nstiff,
        stiff_AR, b, h, rho0, gamma
    )

    def objective(x):
        return np.sum(np.array(gamma_rho0_resid(x))**2)
    
    # can be multiple solutions in AR, h_s inputs for rho0, gamma
    # this helps prevent highly stiffened solutions for low gamma, high AR
    gamma = np.exp(target_log_gamma - 1.0)
    # if gamma < 5: 
    #     ub_log_rho = np.log(rho0 * 1.5)
    # else:
    #     ub_log_rho = np.log(rho0 * 2.5)
    #     # ub_log_rho = np.log(rho0 * 3.5)

    # get initial guess
    # -----------------
    min_err = np.inf
    for log_hw in np.linspace(-3, 1.0, 50):
        for log_rho in np.linspace(-3, 3.0, 40):
            xhat = [log_hw, log_rho]
            myresid = gamma_rho0_resid([log_hw, log_rho])
            resid_norm = np.linalg.norm(np.array(myresid))
            # print(f"{xhat=} {resid_norm=}")
            if resid_norm < min_err:
                if nstiff == 0:
                    guess_x0 = (-7, log_rho)
                else:
                    guess_x0 = (log_hw, log_rho)
                min_err = resid_norm
    if comm.rank == 0: print(f"{guess_x0=} {min_err=}")

    print(f"ant opt: {comm.rank} checkpt1", flush=True)

    # now solve in 10x10 different spots (to try and find the multiple unique solutions)
    log_hw_vec = np.linspace(-3, 2.0, 10); dlog_hw = np.diff(log_hw_vec)[0]
    log_rho_vec = np.linspace(-3, 3.0, 10); dlog_rho = np.diff(log_rho_vec)[0]
    for resid_tol in [1e-2, 1e-1, 1e0]: # just because sometimes a bit too tight for the minimizer
        solns = []
        for log_hw_lb in log_hw_vec:
            for log_rho_lb in log_rho_vec:
                bounds = [(log_hw_lb, log_hw_lb + dlog_hw), (log_rho_lb, log_rho_lb + dlog_rho)]
                # print(f"{bounds=}")
                initial_guess = guess_x0
                # sol = root(gamma_rho0_resid, initial_guess, method='trust-constr', bounds=bounds)
                sol = minimize(objective, initial_guess, bounds=bounds, method='trust-constr')
                xopt = sol.x
                myresid = gamma_rho0_resid(xopt)
                resid_norm = np.linalg.norm(myresid)
                # print(f"{xopt=} {resid_norm=}")
                solved = resid_norm < resid_tol
                if solved:
                    solns += [xopt]
        if len(solns) > 0: break
                

    print(f"ant opt: {comm.rank} checkpt2", flush=True)
    if comm.rank == 0:
        print(f"ant finding method: {solns}", flush=True)

    if plot_debug:
        import matplotlib.pyplot as plt
        log_hw = np.linspace(-3.0, 2.0, 50)
        log_rho = np.linspace(-2, 2, 40)
        LOG_HW, LOG_RHO = np.meshgrid(log_hw, log_rho)
        RESID = np.zeros((40,50))
        for i, my_log_hw in enumerate(log_hw):
            for j, my_log_rho in enumerate(log_rho):
                RESID[j,i] = np.linalg.norm(np.array(gamma_rho0_resid([my_log_hw, my_log_rho])))
        if comm.rank == 0:
            fig, ax = plt.subplots(1,1)
            contour = ax.contourf(LOG_HW, LOG_RHO, RESID, levels=20, vmin=0.0, vmax=2.0)
            cbar = plt.colorbar(contour) 
            # cbar.set_clim(0.0, 1.0)
            plt.xlabel("log hw")
            plt.ylabel("log_rho")
            plt.show()

    # now select xopt form min first arg of the solns
    hws = [xopt[0] for xopt in solns]
    # ind = np.argmin(np.array(hws))
    # ind = hws[hs_ind]
    xopt = solns[hs_ind]

    h_w = 10**(xopt[0])
    AR = 10**(xopt[1])
    a = b * AR

    print(f"ant opt: {comm.rank} checkpt3", flush=True)

    if comm.rank == 0:
        print(f"soln of R(xhat): {h_w=} {AR=}", flush=True)

    return h_w, AR, a