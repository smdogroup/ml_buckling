import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import cm
import niceplots
plt.style.use(niceplots.get_style())

fig = plt.figure("3d-GP", figsize=(14, 9))
ax = plt.axes(projection="3d", computed_zorder=False)

"""
data from 1_unstiffened_panels/2_train_GP_model_v1.py script
"""

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
Dstar_bins = [[0.25 * i, 0.25 * (i + 1)] for i in range(7)]
ibin = 0

slender_bin1 = slender_bins[ibin]
slender_bin = [np.log(slender_bin1[0]), np.log(slender_bin1[1])]
# slender_bin = [bin[0], bin[1]]
avg_log_slender = 0.5 * (slender_bin[0] + slender_bin[1])
mask1 = np.logical_and(slender_bin[0] <= X[:, 2], X[:, 2] <= slender_bin[1])

fig = plt.figure("3d-GP", figsize=(14, 9))
ax = plt.axes(projection="3d", computed_zorder=False)

# plot data in certain range of the training set
for iDstar, Dstar_bin in enumerate(Dstar_bins):
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
        s=40,
        edgecolors="black",
        zorder=1 + iDstar,
    )

# plot the model curve
# Creating plot
face_colors = cm.jet(KMIN / 10.0)
ax.plot_surface(
    DSTAR,
    AR,
    KMIN,
    antialiased=False,
    facecolors=face_colors,
    alpha=0.4,
    zorder=1,
)

# save the figure
# ax.set_xlabel(r"$\xi =(D_{12}^p + 2 D_{66}^p)/(\sqrt{D_{11}^p D_{22}^p})$", fontsize=18)
# ax.set_ylabel(r"$\rho_0 = a/b \cdot \sqrt[4]{D_{22}^p/D_{11}^p}$", fontsize=18)
ax.set_xlabel(r"$\xi$", fontsize=24)
ax.set_ylabel(r"$\rho_0$", fontsize=24)
ax.set_zlabel(r"$N_{11,cr}^*$", fontsize=20)
ax.set_ylim3d(0.0, 10.0)
ax.set_zlim3d(0.0, 10.0)
ax.view_init(elev=20, azim=20, roll=0)
plt.gca().invert_xaxis()
# plt.title(f"b/h in [{bin[0]},{bin[1]}]")
# plt.show()
plt.savefig(f"3d-slender-{ibin}.svg", dpi=400)
plt.savefig(f"3d-slender-{ibin}.png", dpi=400)
plt.close("3d-GP")