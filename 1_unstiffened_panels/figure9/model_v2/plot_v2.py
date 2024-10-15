import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import cm
import niceplots
plt.style.use(niceplots.get_style())
from matplotlib.ticker import FuncFormatter
import ml_buckling as mlb

fig = plt.figure("3d-GP", figsize=(10,7))
ax = plt.axes(projection="3d", computed_zorder=False)

"""
data from 1_unstiffened_panels/2_train_GP_model_v1.py script
"""

# reload the data
ibin = 0 #1
data = np.load(f"array-data{ibin}.npz")
X = data["X"]; Y = data["Y"]
DSTAR, AR, KMIN = data["DSTAR"], data["AR"], data["KMIN"]

slender_bins = [
    [0.0, 0.6],
    [0.6, 1.0],
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.5],
]
Dstar_bins = [[0.25 * i, 0.25 * (i + 1)] for i in range(1,7)]

# slender_bin = [np.log(bin[0]), np.log(bin[1])]
slender_bin = slender_bins[ibin]
avg_log_slender = 0.5 * (slender_bin[0] + slender_bin[1])
mask1 = np.logical_and(slender_bin[0] <= X[:, 2], X[:, 2] <= slender_bin[1])

fig = plt.figure("3d-GP", figsize=(10, 7))
ax = plt.axes(projection="3d", computed_zorder=False)

colors = mlb.six_colors2

# plot data in certain range of the training set
for iDstar, Dstar_bin in enumerate(Dstar_bins[::-1]):
    log_Dstar_bin = np.log(1.0 + np.array(Dstar_bin))
    mask2 = np.logical_and(
        log_Dstar_bin[0] <= X[:, 0], X[:, 0] <= log_Dstar_bin[1]
    )
    avg_Dstar = 0.5 * (log_Dstar_bin[0] + log_Dstar_bin[1])

    mask = np.logical_and(mask1, mask2)
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

# plot the model curve
# Creating plot
face_colors = cm.Grays(0.5 * KMIN / KMIN)
ax.plot_surface(
    DSTAR,
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

xticks = ax.get_xticks()
yticks = ax.get_yticks()
zticks = ax.get_zticks()

ax.set_xticklabels(xticks, fontweight='bold', fontsize=12)
ax.set_yticklabels(yticks, fontweight='bold', fontsize=12)
ax.set_zticklabels(zticks, fontweight='bold', fontsize=12)

# Function to format tick labels
def format_ticks(value, tick_number):
    return f'{value:.1f}'  # Format to 1 decimal places

# Set tick formatting
ax.xaxis.set_major_formatter(FuncFormatter(format_ticks))
ax.yaxis.set_major_formatter(FuncFormatter(format_ticks))
ax.zaxis.set_major_formatter(FuncFormatter(format_ticks))

# save the figure
ax.set_xlabel(r"$\mathbf{ln(1+\xi)}$", fontsize=16, fontweight='bold')
ax.set_ylabel(r"$\mathbf{ln(\rho_0)}$", fontsize=16, fontweight='bold')
ax.set_zlabel(r"$\mathbf{ln(N_{11,cr}^*)}$", fontsize=16, fontweight='bold')
ax.set_ylim3d(np.log(0.1), np.log(10.0))
ax.set_zlim3d(0.0, np.log(30.0))
ax.view_init(elev=20, azim=25, roll=0)
plt.gca().invert_xaxis()
# plt.title(f"b/h in [{bin[0]},{bin[1]}]")
# plt.show()
plt.savefig(f"plotv2-{ibin}.png", dpi=400)
plt.close("3d-GP")