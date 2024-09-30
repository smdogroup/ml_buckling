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
    colors = mlb.six_colors2 #[::-1]

    # # plot the FEA global modes
    global_mask = (mode == "global")[:,0]
    # print(f"{global_mask=} {global_mask.shape=}")
    if np.sum(global_mask) > 0:
        plt.plot(
            np.log(1.0+gamma[global_mask]),
            xslope[global_mask],
            linestyle='-',
            marker='o',
            label='x-ndslope',
            color=colors[0],
            alpha=1.0,
            markersize=6.0,
            zorder=0
            #markersize=6.0 - 1.0 *  istiff,
        )

        plt.plot(
            np.log(1.0+gamma[global_mask]),
            yslope[global_mask],
            linestyle='-',
            marker='o',
            label='y-ndslope',
            color=colors[1],
            alpha=1.0,
            markersize=6.0,
            zorder=1
            #markersize=6.0 - 1.0 *  istiff,
        )
        
        N11_ratio = eig_FEA / eig_CF
        plt.plot(
            np.log(1.0+gamma[global_mask]),
            N11_ratio[global_mask],
            linestyle='-',
            marker='o',
            label=r"$N_{11,FEA}/N_{11,CF}$",
            color=colors[2],
            alpha=1.0,
            markersize=6.0,
            zorder=2
            #markersize=6.0 - 1.0 *  istiff,
        )


    # rho0_FEA[::2], N11_FEA[::2] (when there were 100 points and wanted only 50 of them)

    small_size = 16
    large_size = 20

    # finish making the plot
    plt.legend(prop={'size' : small_size, 'weight' : 'bold'})
    # plt.title(r"$\gamma = 11.25$")
    plt.xlabel(r"$\mathbf{\ln(1+\gamma)}$", fontsize=large_size, fontweight='bold')
    plt.ylabel(r"$\mathbf{ND vals}$", fontsize=large_size, fontweight='bold')
    # plt.xscale('log')
    # plt.yscale('log')

    plt.xticks(fontsize=small_size, fontweight='bold')
    plt.yticks(fontsize=small_size, fontweight='bold')

    plt.margins(x=0.05, y=0.05)

    plt.savefig("9-bc-gamma.png", dpi=400)
    plt.savefig("9-bc-gamma.svg", dpi=400)
