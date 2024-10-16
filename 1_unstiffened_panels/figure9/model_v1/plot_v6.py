import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import cm
import niceplots
from matplotlib.ticker import FuncFormatter

"""
data from 1_unstiffened_panels/2_train_GP_model_v1.py script
"""

plt.style.use(niceplots.get_style())
import ml_buckling as mlb

fig = plt.figure("3d-GP", figsize=(10,7))
ax = plt.axes(projection="3d", computed_zorder=False)

# reload the data
ibin = 1 #1
data = np.load(f"array-data{ibin}.npz")
X = data["X"]; Y = data["Y"]
DSTAR, AR, KMIN = data["DSTAR"], data["AR"], data["KMIN"]

slender_bins = [
    [10.0, 20.0],
    [20.0, 50.0],
    [50.0, 100.0],
    [100.0, 200.0],
]  # [5.0, 10.0],
Dstar_bins = [[0.25 * i, 0.25 * (i + 1)] for i in range(1, 7)]
ibin = 0

slender_bin1 = slender_bins[ibin]
slender_bin = [np.log(slender_bin1[0]), np.log(slender_bin1[1])]
# slender_bin = [bin[0], bin[1]]
avg_log_slender = 0.5 * (slender_bin[0] + slender_bin[1])
mask1 = np.logical_and(slender_bin[0] <= X[:, 2], X[:, 2] <= slender_bin[1])

fig = plt.figure("3d-GP", figsize=(14, 9))
ax = plt.axes(projection="3d", computed_zorder=False)

colors = mlb.six_colors2

# plot data in certain range of the training set
for iDstar, Dstar_bin in enumerate(Dstar_bins[::-1]):
    mask2 = np.logical_and(Dstar_bin[0] <= X[:, 0], X[:, 0] <= Dstar_bin[1])
    avg_Dstar = 0.5 * (Dstar_bin[0] + Dstar_bin[1])
    mask3 = Y[:, 0] < 10.0

    mask = np.logical_and(mask1, mask2)
    mask = np.logical_and(mask, mask3)
    if np.sum(mask) == 0:
        continue
    X_in_range = X[mask, :]
    Y_in_range = Y[mask, :]
    ax.scatter(
        X_in_range[:, 0],
        X_in_range[:, 1],
        Y_in_range[:, 0],
        s=30,
        color=colors[iDstar],
        edgecolors="black",
        zorder=1 + iDstar,
    )

# just clip the KMIN curve manually with white box over it (KMIN > 10 data)
# in drawio

# plot the model curve
# Creating plot
# face_colors = cm.jet(KMIN / 10.0)
face_colors = cm.Grays(0.4 * KMIN / KMIN)
ax.plot_surface(
    DSTAR,
    AR,
    KMIN,
    antialiased=True,
    facecolors=face_colors,
    alpha=0.5,
    # edgecolor='none',
    linewidth=0.3,
    # edgecolor='lightgrey',
    shade=True,
    zorder=1,
)

xticks = ax.get_xticks()
yticks = ax.get_yticks()
zticks = ax.get_zticks()

ax.set_xticklabels(xticks, fontweight='bold', fontsize=16)
ax.set_yticklabels(yticks, fontweight='bold', fontsize=16)
ax.set_zticklabels(zticks, fontweight='bold', fontsize=16)

# Function to format tick labels
def format_ticks(value, tick_number):
    return f'{value:.1f}'  # Format to 1 decimal places

# Set tick formatting
ax.xaxis.set_major_formatter(FuncFormatter(format_ticks))
ax.yaxis.set_major_formatter(FuncFormatter(format_ticks))
ax.zaxis.set_major_formatter(FuncFormatter(format_ticks))

ax.grid(False)
ax.xaxis.pane.set_edgecolor('black')
ax.yaxis.pane.set_edgecolor('black')
ax.zaxis.pane.set_edgecolor('black')

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# save the figure
# ax.set_xlabel(r"$\xi =(D_{12}^p + 2 D_{66}^p)/(\sqrt{D_{11}^p D_{22}^p})$", fontsize=18)
# ax.set_ylabel(r"$\rho_0 = a/b \cdot \sqrt[4]{D_{22}^p/D_{11}^p}$", fontsize=18)
ax.set_xlabel(r"$\mathbf{\xi}$", fontsize=24, fontweight='bold')
ax.set_ylabel(r"$\mathbf{\rho_0}$", fontsize=24, fontweight='bold')
ax.set_zlabel(r"$\mathbf{N_{11,cr}^*}$", fontsize=20, fontweight='bold')
ax.set_ylim(0.0, 10.0)
ax.set_zlim(0.0, 10.0)
ax.view_init(elev=20, azim=25, roll=0)
plt.gca().invert_xaxis()
# plt.title(f"b/h in [{bin[0]},{bin[1]}]")
# plt.show()
plt.savefig(f"3d-slender-{ibin}-v6.svg", dpi=400)
plt.savefig(f"3d-slender-{ibin}-v6.png", dpi=400)
plt.close("3d-GP")