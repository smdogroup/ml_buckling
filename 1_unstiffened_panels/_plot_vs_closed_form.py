import numpy as np
import pandas as pd

# # allow bold math
# import matplotlib as mpl
# rc_fonts = {
#     "text.usetex": True,
#     'text.latex.preview': True, # Gives correct legend alignment.
#     'mathtext.default': 'regular',
#     'text.latex.preamble': [r"""\usepackage{bm}"""],
# }
# mpl.rcParams.update(rc_fonts)
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
con.setCFShearMode(1)
con.setKSWeight(20.0)

"""
This time I'll try a Gaussian Process model to fit the axial critical load surrogate model
Inputs: D*, a0/b0, ln(b/h)
Output: k_x0
"""
# parse the arguments
parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--load", type=str)

args = parent_parser.parse_args()

assert args.load in ["Nx", "Nxy", "axial", "shear"]
args.BC = "SS"

print(f"args.load = {args.load}")
if args.load in ["Nx", "axial"]:
    load = "Nx"
else:
    load = "Nxy"
BC = args.BC
plot_log = load == "Nxy"
# plot_log = False

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
zeta = X[:, 2]
lam = Y[:, 0]

_image = image.imread(f"images/{load_prefix}-{BC}-mode.png")

# colors = plt.cm.jet(np.linspace(0, 1, len(xi_bins)))
# five color custom color map
# colors = mlb.five_colors9[::-1] + ["b"]
colors = mlb.six_colors2  # [::-1]

# plot only the most slender data
slender_bin = [0.0, 0.4]
fig, ax = plt.subplots(figsize=(10, 7))
slender_mask = np.logical_and(slender_bin[0] <= zeta, zeta <= slender_bin[1])

n = 100
rho0_vec = np.linspace(0.1, 10.0, n)
Ncr_vec = np.zeros((n,), dtype=dtype)

# plt.text(x=3.6, y=8.7, s=r"$\xi = \frac{D_{12}^p + D_{66}^p}{\sqrt{D_{11}^p D_{22}^p}}$", fontsize=24)

# # also plot the analytic shear surrogate
# if args.load == "Nxy":
#     con.setCFShearMode(1)

#     for ixi, xi_bin in enumerate(xi_bins[::-1]):
#         # convert from log(xi) to xi here
#         xi_mask = np.logical_and(xi_bin[0] <= np.exp(xi) - 1.0, np.exp(xi) - 1.0 <= xi_bin[1])
#         avg_xi = 0.5 * (xi_bin[0] + xi_bin[1])

#         mask = np.logical_and(xi_mask, slender_mask)
#         if np.sum(mask) == 0:
#             print(f"nothing in mask")
#             continue

#         # plot the closed-form solution
#         for i, _rho0 in enumerate(rho0_vec):
#             Ncr_vec[i] = con.nondimCriticalGlobalShearLoad(_rho0, avg_xi, 0.0)

#         plt.plot(
#             rho0_vec,
#             Ncr_vec,
#             "--",
#             color=colors[ixi],
#         )

#     con.setCFShearMode(1)

for ixi, xi_bin in enumerate(xi_bins[::-1]):
    # convert from log(xi) to xi here
    xi_mask = np.logical_and(
        xi_bin[0] <= np.exp(xi) - 1.0, np.exp(xi) - 1.0 <= xi_bin[1]
    )
    avg_xi = 0.5 * (xi_bin[0] + xi_bin[1])

    mask = np.logical_and(xi_mask, slender_mask)
    if np.sum(mask) == 0:
        print(f"nothing in mask")
        continue

    # plot the closed-form solution
    if args.load in ["Nx", "axial"]:
        for i, _rho0 in enumerate(rho0_vec):
            Ncr_vec[i] = con.nondimCriticalGlobalAxialLoad(_rho0, avg_xi, 0.0)
    else:
        for i, _rho0 in enumerate(rho0_vec):
            Ncr_vec[i] = con.nondimCriticalGlobalShearLoad(_rho0, avg_xi, 0.0)

    if plot_log:
        plt.plot(
            np.log(rho0_vec),
            np.log(Ncr_vec),
            color=colors[ixi],
        )
    else:
        plt.plot(
            rho0_vec,
            Ncr_vec,
            color=colors[ixi],
        )

    # plot the raw data
    if plot_log:
        plt.plot(
            rho0[mask],
            lam[mask],
            "o",
            color=colors[ixi],
            label=r"$\xi\ in\ [" + f"{xi_bin[0]},{xi_bin[1]}" + r"]$",
            markersize=6.5,  # 6.5
        )
    else:
        plt.plot(
            np.exp(rho0[mask]),
            np.exp(lam[mask]),
            "o",
            color=colors[ixi],
            label=r"$\xi\ in\ [" + f"{xi_bin[0]},{xi_bin[1]}" + r"]$",
            markersize=5,  # 6.5
        )


legend1 = plt.legend(fontsize=20, loc="upper right")
if not plot_log:
    plt.xlabel(
        r"$\mathbf{\rho_0}$", fontsize=24, fontweight='bold'
    )
else:
    plt.xlabel(r"$\ln( \mathbf{\rho_0} )$", fontsize=28, fontweight='bold')
# plt.xticks(fontsize=18)
# if args.load == "Nx":
#     plt.ylabel(r"$N_{11,cr}^* = N_{11,cr} \cdot \frac{b^2}{\pi^2 \sqrt{D_{11}^p D_{22}^p}}$", fontsize=24)
# else: # "Nxy"
#     plt.ylabel(r"$N_{12,cr}^* = N_{12,cr} \cdot \frac{b^2}{\pi^2 \sqrt[4]{D_{11}^p (D_{22}^p)^3}}$", fontsize=24)
if not plot_log:
    if args.load in ["Nx", "axial"]:
        plt.ylabel(r"$\mathbf{N_{11,cr}^*}$", fontsize=28, fontweight='bold')
    else:  # "Nxy"
        plt.ylabel(r"$N_{12,cr}^*$", fontsize=28, fontweight='bold')
else:
    if args.load in ["Nx", "axial"]:
        plt.ylabel(r"$\ln(N_{11,cr}^*)$", fontsize=28, fontweight='bold')
    else:  # "Nxy"
        plt.ylabel(r"$\ln(\mathbf{N_{12,cr}^*} )$", fontsize=28, fontweight='bold')
# plt.yticks(fontsize=18)
plt.margins(x=0.02, y=0.02)
if not plot_log:
    plt.xlim(0.0, 5.0)
    # plt.xlim(0.0, 10.0)
    plt.ylim(0.0, 20.0)
    # plt.ylim(0.0, 50.0)
    # plt.yscale('log')
else:
    # pass
    plt.xlim(-2.0, 2.3)
    plt.ylim(1.0, 5.5)
    # plt.ylim(np.min(np.exp(lam[slender_mask])), np.max(np.exp(lam[slender_mask])))

# plot black dots out of axis limits
do_this = True
if do_this:
    (l1,) = plt.plot(np.nan, 0, "ko", label="FEA model")
    # l2, = plt.plot([-1e4, -1e4+1], [0,0], "k--", label="analytic-surrogate")
    (l3,) = plt.plot([np.nan, np.nan], [1, 1], "k-", label="closed-form")
    plot_lines = [l1, l3]
    # plt.hold(True)
    # legend2 = plt.legend(plot_lines, ["FEA model", "least-squares", "closed-form"])#, loc="best", bbox_to_anchor=(3.0, 18.0))
    legend2 = plt.legend(
        plot_lines, ["FEA model", "closed-form"], prop={'size':24, 'weight':'bold'},
    )  # , loc="best", bbox_to_anchor=(3.0, 18.0))
    # plt.gca().add_artist(legend1)

# if plot_log:
# plt.xscale('log')
# plt.yscale('log')

plt.xticks(fontsize=24, fontweight='bold')
plt.yticks(fontsize=24, fontweight='bold')


# plt.show()
plt.savefig(
    os.path.join(sub_sub_data_folder, f"{args.load}-vs-closed-form.svg"), dpi=400
)
plt.savefig(
    os.path.join(sub_sub_data_folder, f"{args.load}-vs-closed-form.png"), dpi=400
)
plt.close("all")
