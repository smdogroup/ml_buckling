import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import niceplots, os
import argparse
from matplotlib import cm
import shutil
from matplotlib.offsetbox import (
    OffsetImage,
    AnnotationBbox,
)  # The OffsetBox is a simple container artist.
import matplotlib.image as image

# The child artists are meant to be drawn at a relative position to its #parent.


"""
This time I'll try a Gaussian Process model to fit the axial critical load surrogate model
Inputs: D*, a0/b0, ln(b/h)
Output: k_x0
"""

load = "Nx"
BC = "SS"

# load the Nxcrit dataset
load_prefix = "Nxcrit" if load == "Nx" else "Nxycrit"
csv_filename = f"{load_prefix}_{BC}"
print(f"csv filename = {csv_filename}")
df = pd.read_csv("_data/" + csv_filename + ".csv")

# extract only the model columns
# TODO : if need more inputs => could maybe try adding log(E11/E22) in as a parameter?
# or also log(E11/G12)
X = df[["x0", "x1", "x2"]].to_numpy()
Y = df["y"].to_numpy()
Y = np.reshape(Y, newshape=(Y.shape[0], 1))

print(f"Monte Carlo #data = {X.shape[0]}")
N_data = X.shape[0]

n_train = int(0.9 * N_data)

# REMOVE THE OUTLIERS in local 4d regions
# loop over different slenderness bins
slender_bins = [
    [10.0, 20.0],
    [20.0, 50.0],
    [50.0, 100.0],
    [100.0, 200.0],
]  # [5.0, 10.0],
xi_bins = [[0.25 * i, 0.25 * (i + 1)] for i in range(1, 7)]
# added smaller and larger bins here cause we miss some of the outliers near the higher a0/b0 with less data
aff_AR_bins = (
    [[0.5 * i, 0.5 * (i + 1)] for i in range(4)]
    + [[1.0 * i, 1.0 * (i + 1)] for i in range(2, 5)]
    + [[2.0, 10.0]]
)

# make a folder for the model fitting
data_folder = os.path.join(os.getcwd(), "_plots")
sub_data_folder = os.path.join(data_folder, csv_filename)
sub_sub_data_folder = os.path.join(sub_data_folder, "data")
for ifolder, folder in enumerate(
    [
        data_folder,
        sub_data_folder,
        sub_sub_data_folder,
    ]
):
    if not os.path.exists(folder):
        os.mkdir(folder)

plt.style.use(niceplots.get_style())
xi = X[:, 0]
rho0 = X[:, 1]
slenderness = X[:, 2]
lam = Y[:, 0]

colors = plt.cm.jet(np.linspace(0, 1, len(slender_bins)))

slender_bins = slender_bins[::-1]

# now plot by xi
for ixi, xi_bin in enumerate(xi_bins):
    fig, ax = plt.subplots(figsize=(10, 7))
    xi_mask = np.logical_and(xi_bin[0] <= np.exp(xi), np.exp(xi) <= xi_bin[1])

    plt.arrow(x=1.4, y=14, dx=0, dy=-6, facecolor="b", width=0.01)
    plt.text(x=1.4, y=14.5, s="Mode#", horizontalalignment="center")

    for iAR in range(1, 4):
        plt.arrow(x=iAR, y=7, dx=0, dy=-1.8, facecolor="k", width=0.01)
        plt.text(
            x=iAR, y=14, s=r"$\rho = " + str(iAR) + r"$", horizontalalignment="center"
        )
        for imode in range(1, 4):
            _image = image.imread(f"images/NxSS-{iAR}_{imode}.png")

            # plt.text(2+dx, 18+dy, text, horizontalalignment="center")
            zoom = 0.05
            if iAR == 3:
                zoom = 0.1
            imagebox = OffsetImage(
                _image, zoom=zoom
            )  # Annotation box for solar pv logo
            # Container for the imagebox referring to a specific position *xy*.
            ab = AnnotationBbox(
                imagebox, (iAR, 8 + 4.6 - 2.4 * (imode - 1)), frameon=False
            )
            ax.add_artist(ab)

    # get xy coords of point at rho0 = 1.0
    mask1 = np.logical_and(np.log(1.4) <= xi, xi <= np.log(1.7))
    mask2 = np.logical_and(np.log(0.9) <= rho0, rho0 <= np.log(1.1))
    mask = np.logical_and(mask1, mask2)
    lam_near = np.exp(lam[mask][0])
    rho0_near = np.exp(rho0[mask][0])

    # plt.arrow(x=2+dx, y=13+dy, dx=-1.0-dx, dy=lam_near - 13-dy, width=0.01, facecolor="k")

    for islender, slender_bin in enumerate(slender_bins):
        slender_mask = np.logical_and(
            slender_bin[0] <= np.exp(slenderness), np.exp(slenderness) <= slender_bin[1]
        )

        mask = np.logical_and(xi_mask, slender_mask)
        if np.sum(mask) == 0:
            continue

        plt.plot(
            np.exp(rho0[mask]),
            np.exp(lam[mask]),
            "o",
            color=colors[islender],
            zorder=len(slender_bins) - islender,
            label=r"$b/h\ in\ [" + f"{slender_bin[0]},{slender_bin[1]}" + r"]$",
        )

    plt.legend()
    plt.xlabel(r"$\rho_0$")
    plt.ylabel(r"$\lambda_{min}^*$")
    plt.margins(x=0.02, y=0.02)
    plt.xlim(0.0, 4.0)
    plt.ylim(0.0, 15.0)
    # plt.show()
    plt.savefig(os.path.join(sub_sub_data_folder, f"xi{ixi}.png"), dpi=400)
    plt.close("all")
