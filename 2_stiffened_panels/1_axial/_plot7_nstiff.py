import pandas as pd
import matplotlib.pyplot as plt
import niceplots
import ml_buckling as mlb
import numpy as np

"""
plotting script for previously saved data from the script
3_timoshenko_dataset.py
"""

if __name__ == "__main__":

    df = pd.read_csv("7-nstiff-compare.csv")
    X = df[["rho0", "gamma", "nstiff"]].to_numpy()
    Y = df[["eig_CF", "eig_FEA"]].to_numpy()

    rho0 = X[:, 0]
    gamma = X[:, 1]
    nstiff = X[:,2]
    eig_CF = Y[:, 0]
    eig_FEA = Y[:, 1]

    nstiff_unique = np.unique(nstiff)

    plt.style.use(niceplots.get_style())

    plt.figure("this")
    colors = mlb.six_colors2[::-1]

    for istiff, nstiff_val in enumerate(nstiff_unique[::-1]):

        nstiff_mask = nstiff == nstiff_val

        # plot the FEA
        # rho0_FEA = rho0[gamma_mask]
        # N11_FEA = eig_FEA[gamma_mask]
        # plt.plot(rho0_FEA, N11_FEA, "o", label=r"$\gamma = " + f"{gamma_val:.1f}" + r"$", color=colors[igamma], alpha=0.9)

        # plot the closed-form
        if nstiff_val == 1:
            rho0_CF = rho0[nstiff_mask]
            N11_CF = eig_CF[nstiff_mask]
            plt.plot(
                np.log(rho0_CF),
                np.log(N11_CF),
                "-",
                label=None,
                color='k',
                linewidth=2.5,
            )

        # # plot the FEA
        rho0_FEA = rho0[nstiff_mask]
        N11_FEA = eig_FEA[nstiff_mask]
        # plot every other point for now when there were 100 per line (50 per line is a good density where can see points and line separately)
        plt.plot(
            np.log(rho0_FEA),
            np.log(N11_FEA),
            "o",
            label=r"$N_s= " + f"{int(nstiff_val):d}" + r"$",
            color=colors[istiff],
            alpha=0.5,
            markersize=5,
        )
        # rho0_FEA[::2], N11_FEA[::2] (when there were 100 points and wanted only 50 of them)

    # finish making the plot
    plt.legend()
    # plt.title(r"$\gamma = 11.25$")
    plt.xlabel(r"$\log(\rho_0)$")
    plt.ylabel(r"$\log(N_{11,cr}^*)$")
    # plt.xscale('log')
    # plt.yscale('log')
    plt.margins(x=0.05, y=0.05)
    # plt.show()
    plt.savefig("7-nstiff-compare.png", dpi=400)
    plt.savefig("7-nstiff-compare.svg", dpi=400)
