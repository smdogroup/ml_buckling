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
Dstar_bins = [[0.25 * i, 0.25 * (i + 1)] for i in range(7)]

# slender_bin = [np.log(bin[0]), np.log(bin[1])]
slender_bin = slender_bins[ibin]
avg_log_slender = 0.5 * (slender_bin[0] + slender_bin[1])
mask1 = np.logical_and(slender_bin[0] <= X[:, 2], X[:, 2] <= slender_bin[1])

fig = plt.figure("3d-GP", figsize=(14, 9))
ax = plt.axes(projection="3d", computed_zorder=False)

# plot data in certain range of the training set
for iDstar, Dstar_bin in enumerate(Dstar_bins):
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
        s=40,
        edgecolors="black",
        zorder=1 + iDstar,
    )

# plot the model curve
# Creating plot
face_colors = cm.jet((KMIN - 0.8) / np.log(10.0))
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
ax.set_xlabel(r"$\log(1+\xi)$")
ax.set_ylabel(r"$log(\rho_0)$")
ax.set_zlabel(r"$log(N_{11,cr}^*)$")
ax.set_ylim3d(np.log(0.1), np.log(10.0))
ax.set_zlim3d(0.0, np.log(30.0))
ax.view_init(elev=20, azim=20, roll=0)
plt.gca().invert_xaxis()
# plt.title(f"b/h in [{bin[0]},{bin[1]}]")
# plt.show()
plt.savefig(f"plotv1-{ibin}.png", dpi=400)
plt.close("3d-GP")