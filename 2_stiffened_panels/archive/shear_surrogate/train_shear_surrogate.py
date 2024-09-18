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
import ml_buckling as mlb
from tacs import TACS, constitutive

DEG2RAD = np.pi / 180.0

dtype = TACS.dtype

# Create the orthotropic layup
ortho_prop = constitutive.MaterialProperties(
    rho=1550,
    specific_heat=921.096,
    E1=54e3,
    E2=18e3,
    nu12=0.25,
    G12=9e3,
    G13=9e3,
    G23=9e3,
    Xt=2410.0,
    Xc=1040.0,
    Yt=73.0,
    Yc=173.0,
    S12=71.0,
    alpha=24.0e-6,
    kappa=230.0,
)
ortho_ply = constitutive.OrthotropicPly(1e-3, ortho_prop)

# don't put in any GP models (so using closed-form solutions rn)
con = constitutive.GPBladeStiffenedShellConstitutive(
    panelPly=ortho_ply,
    stiffenerPly=ortho_ply,
    panelLength=2.0,
    stiffenerPitch=0.2,
    panelThick=1.5e-2,
    panelPlyAngles=np.array([0.0, 45.0, 90.0], dtype=dtype) * DEG2RAD,
    panelPlyFracs=np.array([0.5, 0.3, 0.2], dtype=dtype),
    stiffenerHeight=0.075,
    stiffenerThick=1e-2,
    stiffenerPlyAngles=np.array([0.0, 60.0], dtype=dtype) * DEG2RAD,
    stiffenerPlyFracs=np.array([0.6, 0.4], dtype=dtype),
    panelWidth=1.0,
    flangeFraction=0.8,
)
# Set the KS weight really low so that all failure modes make a
# significant contribution to the failure function derivatives
con.setKSWeight(20.0)

"""
This time I'll try a Gaussian Process model to fit the axial critical load surrogate model
Inputs: D*, a0/b0, ln(b/h)
Output: k_x0
"""
# parse the arguments
parent_parser = argparse.ArgumentParser(add_help=False)
args = parent_parser.parse_args()

args.load = "Nxy"
args.BC = "SS"

print(f"args.load = {args.load}")
if args.load in ["Nx", "axial"]:
    load = "Nx"
else:
    load = "Nxy"
BC = args.BC

# load the Nxcrit dataset
csv_filename = "Nxy_unstiffened.csv"
df = pd.read_csv("../data/Nxy_unstiffened.csv")

# extract only the model columns
# TODO : if need more inputs => could maybe try adding log(E11/E22) in as a parameter?
# or also log(E11/G12)
X = df[["x0", "x1", "x2"]].to_numpy()
Y = df["y"].to_numpy()
Y = np.reshape(Y, newshape=(Y.shape[0], 1))

print(f"Monte Carlo #data = {X.shape[0]}")
N_data = X.shape[0]

n_train = int(0.9 * N_data)

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
xi = np.exp(X[:, 0]) - 1.0
rho0 = np.exp(X[:, 1])
gamma = xi * 0.0
zeta = (np.exp(X[:, 2]) - 1.0) / 1000.0
lam = np.exp(Y[:, 0])

# _image = image.imread(f"images/{load_prefix}-{BC}-mode.png")

# colors = plt.cm.jet(np.linspace(0, 1, len(xi_bins)))
# five color custom color map
# colors = mlb.five_colors9[::-1] + ["b"]
colors = mlb.six_colors2  # [::-1]

# plot only the most slender data
slender_bin = [0.0, 0.4]
fig, ax = plt.subplots(figsize=(10, 7))

n = 100
rho0_vec = np.linspace(0.2, 5.0, n)
Ncr_vec = np.zeros((n,), dtype=dtype)


# train the shear surrogate model with
gamma_mask = gamma < 0.01
zeta_mask = zeta < 1e-3
rho0_mask = rho0 > 0.5
mask = np.logical_and(gamma_mask, zeta_mask)
mask = np.logical_and(mask, rho0_mask)
xi1 = xi[mask]
rho01 = rho0[mask]
gamma1 = gamma[mask]
zeta1 = zeta[mask]
lam1 = lam[mask]
npts = lam1.shape[0]

Xlinear = np.zeros((npts, 4))
Ylinear = (lam1 - 0.5 * gamma1).reshape((npts, 1))
# least-squares fit now
Xlinear[:, 0] = 1.0
Xlinear[:, 1] = np.power(rho01, -2.0)
# Xlinear[:,2] = np.power(rho0, -4.0)
Xlinear[:, 2] = xi1
Xlinear[:, 3] = xi1 * np.power(rho01, -2.0)
print(f"Ylinear = {Ylinear}")
print(f"Xlinear = {Xlinear}")
wLS = np.linalg.solve(Xlinear.T @ Xlinear, Xlinear.T @ Ylinear)[:, 0]
print(f"wLS = {wLS}")
# exit()

slender_mask = zeta < 1e-3

plt.text(
    x=3.6,
    y=8.7,
    s=r"$\xi = \frac{D_{12}^p + D_{66}^p}{\sqrt{D_{11}^p D_{22}^p}}$",
    fontsize=24,
)

for ixi, xi_bin in enumerate(xi_bins[::-1]):
    # convert from log(xi) to xi here
    xi_mask = np.logical_and(xi_bin[0] <= xi, xi <= xi_bin[1])
    avg_xi = 0.5 * (xi_bin[0] + xi_bin[1])

    mask = np.logical_and(xi_mask, slender_mask)
    if np.sum(mask) == 0:
        print(f"nothing in mask")
        continue

    # plot the closed-form solution
    # if args.load == "Nx":
    #    for i, _rho0 in enumerate(rho0_vec):
    #        Ncr_vec[i] = con.nondimCriticalGlobalAxialLoad(_rho0, avg_xi, 0.0)
    # else:
    #    for i, _rho0 in enumerate(rho0_vec):
    #        Ncr_vec[i] = con.nondimCriticalGlobalShearLoad(_rho0, avg_xi, 0.0)
    for i, _rho0 in enumerate(rho0_vec):
        Ncr_vec[i] = (
            wLS[0]
            + wLS[1] * _rho0 ** (-2.0)
            + wLS[2] * avg_xi
            + 0.5 * 0.0
            + wLS[3] * avg_xi * _rho0 ** (-2.0)
        )

    plt.plot(
        rho0_vec,
        Ncr_vec,
        color=colors[ixi],
    )

    # plot the raw data
    print(f"rho0[mask] = {rho0[mask]}")
    plt.plot(
        rho0[mask],
        lam[mask],
        "o",
        color=colors[ixi],
        label=r"$\xi\ in\ [" + f"{xi_bin[0]},{xi_bin[1]}" + r"]$",
        markersize=6.5,
    )

plt.legend(fontsize=20, loc="upper right")
plt.xlabel(r"$\rho_0 = \frac{a}{b} \cdot \sqrt[4]{D_{22}^p /D_{11}^p}$", fontsize=24)
plt.xticks(fontsize=18)
# if args.load == "Nx":
#     plt.ylabel(r"$N_{11,cr}^* = N_{11,cr} \cdot \frac{b^2}{\pi^2 \sqrt{D_{11}^p D_{22}^p}}$", fontsize=24)
# else: # "Nxy"
#     plt.ylabel(r"$N_{12,cr}^* = N_{12,cr} \cdot \frac{b^2}{\pi^2 \sqrt[4]{D_{11}^p (D_{22}^p)^3}}$", fontsize=24)
if args.load == "Nx":
    plt.ylabel(r"$N_{11,cr}^*$", fontsize=24)
else:  # "Nxy"
    plt.ylabel(r"$N_{12,cr}^*$", fontsize=24)
plt.yticks(fontsize=18)
plt.margins(x=0.02, y=0.02)
plt.xlim(0.0, 5.0)
# plt.xlim(0.0, 10.0)
plt.ylim(0.0, 20.0)
# plt.show()
plt.savefig("{args.load}-vs-closed-form.svg", dpi=400)
plt.savefig(f"{args.load}-vs-closed-form.png", dpi=400)
plt.close("all")
print(f"wLS = {wLS}")
#
