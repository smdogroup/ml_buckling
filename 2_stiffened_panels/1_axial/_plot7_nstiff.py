import pandas as pd
import matplotlib.pyplot as plt
import niceplots
import ml_buckling as mlb
import numpy as np

"""
plotting script for previously saved data from the script
7_nstiff_compare_dataset.py
"""

if __name__ == "__main__":

    df = pd.read_csv("7-nstiff-compare.csv")
    X = df[["rho0", "gamma", "nstiff", "mode"]].to_numpy()
    Y = df[["eig_CF", "eig_FEA"]].to_numpy()

    rho0 = X[:, 0]
    gamma = X[:, 1]
    nstiff = X[:,2]
    mode = X[:,3]
    eig_CF = Y[:, 0]
    eig_FEA = Y[:, 1]

    nstiff_unique = np.unique(nstiff)

    plt.style.use(niceplots.get_style())

    plt.figure("this")
    colors = mlb.six_colors2 #[::-1]

    nvals = nstiff_unique.shape[0]
    for istiff, nstiff_val in enumerate(nstiff_unique[::-1]):

        nstiff_mask = nstiff == nstiff_val
        # print(f"{nstiff_mask=}")

        # # plot the FEA global modes
        global_mask = np.logical_and(
            nstiff_mask,
            mode == "global"
        )
        rho0_FEA = rho0[global_mask].astype(np.double)
        N11_FEA = eig_FEA[global_mask].astype(np.double)
        # plot every other point for now when there were 100 per line (50 per line is a good density where can see points and line separately)
        if np.sum(global_mask) > 0:
            plt.plot(
                np.log(rho0_FEA),
                np.log(N11_FEA),
                linestyle='-',
                marker="o",
                label=r"$N_s= " + f"{int(nstiff_val):d}" + r"$",
                color=colors[istiff],
                alpha=1.0,
                markersize=5.0,
                zorder=nstiff_val
                # markersize=6.0 - 1.0 *  istiff,
            )

        # # plot the FEA local modes
        local_mask = np.logical_and(
            nstiff_mask,
            mode == "local"
        )
        rho0_FEA = rho0[local_mask].astype(np.double)
        N11_FEA = eig_FEA[local_mask].astype(np.double)
        if np.sum(local_mask) > 0:
            plt.plot(
                np.log(rho0_FEA),
                np.log(N11_FEA),
                linestyle='None',
                marker='^',
                label=None,
                color=colors[istiff],
                alpha=1.0,
                markersize=6.0,
                zorder=nstiff_val
                #markersize=6.0 - 1.0 *  istiff,
            )

        # plot the closed-form
        if istiff == nvals-1:
            rho0_min = np.min(rho0)
            rho0_max = np.max(rho0)
            rho0_CF = np.geomspace(rho0_min, rho0_max, 300)
            N11_CF = 0.0 * rho0_CF
            gamma = 3.0

            for irho0, rho0_val in enumerate(rho0_CF):
                lam_star_global = min(
                    [
                        (1 + gamma) * m1 ** 2 / rho0_val ** 2
                        + rho0_val ** 2 / m1 ** 2
                        + 2 * 1.0
                        for m1 in range(1, 50)
                    ]
                )
                N11_CF[irho0] = lam_star_global

            # N11_CF = np.array(
            #     [axial_load(rho0, gamma=gamma, nstiff=nstiff, prev_dict=None, solve_buckling=False)[1] for rho0 in rho0_CF]
            # )

            # rho0_CF = rho0[nstiff_mask]
            # N11_CF = eig_CF[nstiff_mask]
            plt.plot(
                np.log(rho0_CF),
                np.log(N11_CF),
                "-",
                label="CF",
                color='k',
                linewidth=2.5,
                zorder=0
            )


        # rho0_FEA[::2], N11_FEA[::2] (when there were 100 points and wanted only 50 of them)

    small_size = 16
    large_size = 20

    # finish making the plot
    plt.legend(prop={'size' : small_size, 'weight' : 'bold'})
    # plt.title(r"$\gamma = 11.25$")
    plt.xlabel(r"$\mathbf{\ln(\rho_0)}$", fontsize=large_size, fontweight='bold')
    plt.ylabel(r"$\mathbf{\ln(\overline{N}_{11}^{cr})}$", fontsize=large_size, fontweight='bold')
    # plt.xscale('log')
    # plt.yscale('log')

    plt.xticks(fontsize=small_size, fontweight='bold')
    plt.yticks(fontsize=small_size, fontweight='bold')

    plt.margins(x=0.05, y=0.05)

    plt.savefig("7-nstiff-compare.png", dpi=400)
    plt.savefig("7-nstiff-compare.svg", dpi=400)
