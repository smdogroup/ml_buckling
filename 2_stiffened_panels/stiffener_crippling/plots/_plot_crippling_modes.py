import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import niceplots
import os

df = pd.read_csv("../raw_data/stiffener_crippling.csv")
xi = df["xi"].to_numpy()
slenderness = np.log(df["b/h"].to_numpy())
gen_eps = df["gen_eps"].to_numpy()
affine_AR = df["a0/b0"].to_numpy()
kmin = df["kmin"].to_numpy()

slender_bins = [
    [10.0, 20.0],
    [20.0, 50.0],
    [50.0, 100.0],
    [100.0, 200.0],
]  # [5.0, 10.0],
xi_bins = [[0.25 * i, 0.25 * (i + 1)] for i in range(1, 7)]
# added smaller and larger bins here cause we miss some of the outliers near the higher a0/b0 with less data


plt.style.use(niceplots.get_style())

# iterate over the different slender,D* bins
for ibin, bin in enumerate(slender_bins):
    slender_bin = [np.log(bin[0]), np.log(bin[1])]
    avg_log_slender = 0.5 * (slender_bin[0] + slender_bin[1])
    mask1 = np.logical_and(slender_bin[0] <= slenderness, slenderness <= slender_bin[1])
    if np.sum(mask1) == 0:
        continue

    plt.figure("check model", figsize=(8, 6))
    plt.margins(x=0.05, y=0.05)
    plt.title(f"b/h in [{bin[0]},{bin[1]}]")
    ax = plt.subplot(111)

    for iDstar, Dstar_bin in enumerate(xi_bins[::-1]):
        mask2 = np.logical_and(Dstar_bin[0] <= xi, xi <= Dstar_bin[1])
        avg_Dstar = 0.5 * (Dstar_bin[0] + Dstar_bin[1])

        mask = np.logical_and(mask1, mask2)
        if np.sum(mask) == 0:
            continue

        # plot data in certain range of the training set
        mask = np.logical_and(mask1, mask2)
        X_in_range = affine_AR[mask]
        Y_in_range = kmin[mask]

        colors = plt.cm.jet(np.linspace(0.0, 1.0, len(xi_bins)))

        # plot the raw data and the model in this range
        ax.plot(
            X_in_range,
            Y_in_range,
            "o",
            # markersize=3,
            color=colors[iDstar],
            label=r"$\xi" + f"-[{Dstar_bin[0]},{Dstar_bin[1]}]" + r"$",
        )

    # outside of for loop save the plot
    plt.xlabel(r"$\rho_0$")
    plt.ylabel(r"$\lambda_{min}^*$")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    # plt.xlim(0.0, 10.0)
    # plt.ylim(0.0, 5.0)
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # # Put a legend to the right of the current axis
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f"slender{ibin}.png", dpi=400)
    plt.close("check model")
