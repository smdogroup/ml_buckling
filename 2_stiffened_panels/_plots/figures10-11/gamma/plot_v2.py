import numpy as np
import matplotlib.pyplot as plt
import ml_buckling as mlb
from mpl_toolkits import mplot3d
from matplotlib import cm
import argparse
import niceplots
plt.style.use(niceplots.get_style())

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--load", type=str, default="Nx")
args = parent_parser.parse_args()
assert args.load in ["Nx", "Nxy"]

# reload the data
ibin = 0 #1
data = np.load(f"{args.load}-gamma-data.npz")
X = data["X"]; Y = data["Y"]
GAMMA, AR, KMIN = data["GAMMA"], data["AR"], data["KMIN"]

# bins for the data (in log space)
xi_bins = [[0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1.05]]
rho0_bins = [[-2.5, -1.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 2.5]]
zeta_bins = [[0.0, 0.1], [0.1, 0.5], [0.5, 1.0], [1.0, 2.5]]
# gamma_bins = [[0.0, 0.1], [0.1, 1.0], [1.0, 3.0], [3.0, 5.0]]
# gamma_bins = [[0.0, 0.1], [0.1, 1.0], [1.0, 2.0], [2.0, 3.0]]
gamma_vec = np.geomspace(1+0.0, 1+3.0, 6+1)-1
gamma_bins = [[gamma_vec[i], gamma_vec[i+1]] for i in range(6)]

# 3d plot of rho_0, gamma, lam_star for a particular xi and zeta range
xi_bin = [0.5, 0.7]
xi_mask = np.logical_and(xi_bin[0] <= X[:, 0], X[:, 0] <= xi_bin[1])
avg_xi = 0.5 * (xi_bin[0] + xi_bin[1])

zeta_bin = [0.2, 0.4]
zeta_mask = np.logical_and(zeta_bin[0] <= X[:, 2], X[:, 2] <= zeta_bin[1])
avg_zeta = 0.5 * (zeta_bin[0] + zeta_bin[1])
xi_zeta_mask = np.logical_and(xi_mask, zeta_mask)

plt.figure(f"3d rho_0, gamma, lam_star")
ax = plt.axes(projection="3d", computed_zorder=False)

# colors = plt.cm.jet(np.linspace(0.0, 1.0, len(gamma_bins)))
colors = mlb.six_colors2[::-1]

for igamma, gamma_bin in enumerate(gamma_bins):

    gamma_mask = np.logical_and(
        gamma_bin[0] <= X[:, 3], X[:, 3] <= gamma_bin[1]
    )
    mask = np.logical_and(xi_zeta_mask, gamma_mask)

    rho0_bin = [np.log(0.3), np.log(10.0)]
    rho0_mask = np.logical_and(
        rho0_bin[0] <= X[:,1], X[:,1] <= rho0_bin[1]
    )
    mask = np.logical_and(mask, rho0_mask)

    X_in_range = X[mask, :]
    Y_in_range = Y[mask, :]

    ax.scatter(
        X_in_range[:, 3],
        X_in_range[:, 1],
        Y_in_range[:, 0],
        s=20,
        color=colors[igamma],
        edgecolors="black",
        zorder=2 + igamma,
    )

# plot the model curve
# Creating plot
face_colors = cm.Grays(0.5 * KMIN / KMIN)
ax.plot_surface(
    GAMMA,
    AR,
    KMIN,
    antialiased=True,
    facecolors=face_colors,
    alpha=0.7,
    # edgecolor='none',
    linewidth=0.3,
    # edgecolor='lightgrey',
    shade=True,
    zorder=1,
)

# save the figure
fs1 = 16
ax.set_xlabel(r"$\mathbf{\ln(1+\gamma)}$", fontsize=fs1, fontweight='bold')
ax.set_ylabel(r"$\mathbf{\ln(\rho_0)}$", fontsize=fs1, fontweight='bold')
if args.load == "Nx":
    ax.set_zlabel(r"$\mathbf{\ln(N_{11,cr}^*)}$", fontsize=fs1, fontweight='bold')
else:
    ax.set_zlabel(r"$\mathbf{\ln(N_{12,cr}^*)}$", fontsize=fs1, fontweight='bold')
ax.set_ylim3d(np.log(0.3), np.log(10.0))
# ax.set_zlim3d(0.0, np.log(50.0))
# ax.set_zlim3d(1.0, 3.0)
ax.view_init(elev=20, azim=20, roll=0)
plt.gca().invert_xaxis()

ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.zaxis.set_tick_params(labelsize=14)
# plt.title(f"")
# plt.show()
plt.savefig(f"{args.load}-v2.png", dpi=400)
plt.close(f"3d rho_0, gamma, lam_star")