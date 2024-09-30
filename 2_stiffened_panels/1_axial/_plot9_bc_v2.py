import pandas as pd
import matplotlib.pyplot as plt
import niceplots
import ml_buckling as mlb
import numpy as np

"""
plotting script for previously saved data from the script
9_bc_dataset.py
"""

if __name__ == "__main__":

    df = pd.read_csv("9-bc-gamma-dataset.csv")
    X = df[["x_ndslope", "y_ndslope", "gamma"]].to_numpy()
    Y = df[["eig_CF", "eig_FEA"]].to_numpy()
    mode = df[["mode"]].to_numpy()

    xslope = X[:,0]
    yslope = X[:,1]
    gamma = X[:,2]
    eig_CF = Y[:, 0]
    eig_FEA = Y[:, 1]

    plt.style.use(niceplots.get_style())

    plt.figure("this")
    colors = mlb.six_colors2[::-1]

    gamma_ln = np.log(1.0 + gamma)
    gamma_bins = [[0.0, 0.3], [0.3, 0.6], [0.6, 0.9], [0.9, 1.2], [1.2, 1.5], [1.5, 1.8]]

    # # plot the FEA global modes
    global_mask = (mode == "global")[:,0]
    if np.sum(global_mask) == 0: exit()
    N11_ratio = eig_FEA / eig_CF

    for igamma, gamma_bin in enumerate(gamma_bins[::-1]):
        gamma_mask = np.logical_and(gamma_ln >= gamma_bin[0], gamma_ln < gamma_bin[1])
        gamma_mask = np.logical_and(gamma_mask, global_mask)
        plt.plot(
            xslope[gamma_mask],
            N11_ratio[gamma_mask],
            linestyle='-',
            marker='o',
            label=r"$ln(1+\gamma) \in " + f"[{gamma_bin[0]}, {gamma_bin[1]}]" + r"$",
            color=colors[igamma],
            alpha=1.0,
            markersize=6.0,
            zorder=igamma
            #markersize=6.0 - 1.0 *  istiff,
        )

    # rho0_FEA[::2], N11_FEA[::2] (when there were 100 points and wanted only 50 of them)

    small_size = 14
    large_size = 20

    # finish making the plot
    plt.legend(prop={'size' : small_size, 'weight' : 'bold'})
    # plt.title(r"$\gamma = 11.25$")
    plt.xlabel(r"$\bf{xedge\ nondim\ slope}$", fontsize=large_size, fontweight='bold')
    plt.ylabel(r"$\mathbf{N_{11,FEA} / N_{11,CF}}$", fontsize=large_size, fontweight='bold')
    # plt.xscale('log')
    # plt.yscale('log')

    plt.xticks(fontsize=small_size, fontweight='bold')
    plt.yticks(fontsize=small_size, fontweight='bold')

    plt.margins(x=0.05, y=0.05)

    plt.savefig("9-bc-gamma2.png", dpi=400)
    plt.savefig("9-bc-gamma2.svg", dpi=400)
