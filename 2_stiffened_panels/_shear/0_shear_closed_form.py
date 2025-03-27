import numpy as np
from scipy.optimize import fsolve
import pandas as pd
import ml_buckling as mlb


def shear_closed_form(rho0, xi, gamma):
    """equation for shear closed-form predictions"""

    # high aspect ratio soln
    def high_AR_resid(s2):
        s1 = (1.0 + 2.0 * s2 ** 2 * xi + s2 ** 4 + gamma) ** 0.25
        term1 = s2 ** 2 + s1 ** 2 + xi / 3
        term2 = ((3 + xi) / 9.0 + 4.0 / 3.0 * s1 ** 2 * xi + 4.0 / 3.0 * s1 ** 4) ** 0.5
        return term1 - term2

    s2_bar = fsolve(high_AR_resid, 1.0)[0]
    s1_bar = (1.0 + 2.0 * s2_bar ** 2 * xi + s2_bar ** 4 + gamma) ** 0.25
    print(f"{s1_bar=}, {s2_bar=}")

    N12cr_highAR = (
        (
            1.0
            + gamma
            + s1_bar ** 4
            + 6 * s1_bar ** 2 * s2_bar ** 2
            + s2_bar ** 4
            + 2 * xi * (s1_bar ** 2 + s2_bar ** 2)
        )
        / 2.0
        / s1_bar ** 2
        / s2_bar
    )
    N12cr_lowAR = N12cr_highAR / rho0 ** 2
    return np.max([N12cr_highAR, N12cr_lowAR])


import matplotlib.pyplot as plt
import niceplots

# xi_vec = [0.2, 0.4, 0.6, 0.8, 1.0]
xi_bins = [[0.25 * i, 0.25 * (i + 1)] for i in range(1, 7)]
# xi_vec = [np.min([0.25*i, 0.25*(i+1)]) for i in range(1,5+1)]
# colors = plt.cm.jet(np.linspace(0.0, 1.0, len(xi_vec)))
colors = mlb.six_colors2

# also plot the raw data of unstiffened against the closed-form solution
df = pd.read_csv("../data/Nxy_unstiffened.csv")
X = df[["x0", "x1", "x2"]].to_numpy()
Y = df["y"].to_numpy()
Y = np.reshape(Y, newshape=(Y.shape[0], 1))

plt.style.use(niceplots.get_style())

xi_data = X[:, 0]
rho0_data = X[:, 1]
zeta_data = X[:, 2]
eig_data = Y[:, 0]

# plot the closed-form solutions
rho0_vec = np.geomspace(0.1, 10.0, 100)
for ixi, xi_bin in enumerate(xi_bins[::-1]):
    xi_mask = np.logical_and(
        xi_bin[0] <= np.exp(xi_data) - 1.0, np.exp(xi_data) - 1.0 <= xi_bin[1]
    )
    avg_xi = 0.5 * (xi_bin[0] + xi_bin[1])
    N12cr_vec = np.array([shear_closed_form(rho0, avg_xi, 0.0) for rho0 in rho0_vec])
    # plt.plot(rho0_vec, N12cr_vec, color=colors[ixi], label=None)
    plt.plot(np.log(rho0_vec), np.log(N12cr_vec), color=colors[ixi])

for ixi, xi_bin in enumerate(xi_bins[::-1]):
    xi_mask = np.logical_and(
        xi_bin[0] <= np.exp(xi_data) - 1.0, np.exp(xi_data) - 1.0 <= xi_bin[1]
    )
    avg_xi = 0.5 * (xi_bin[0] + xi_bin[1])
    # zeta_mask = zeta_data < 1.0
    zeta_mask = zeta_data < 0.8
    mask = np.logical_and(xi_mask, zeta_mask)
    # take out of log scale
    # plt.plot(
    #     np.exp(rho0_data[mask]), np.exp(eig_data[mask]), "o", color=colors[ixi],
    #     label=r"$\xi\ in\ [" + f"{xi_bin[0]},{xi_bin[1]}" + r"]$",
    #     markersize=5 #6.5
    # )
    plt.plot(
        rho0_data[mask],
        eig_data[mask],
        "o",
        color=colors[ixi],
        label=r"$\xi\ in\ [" + f"{xi_bin[0]},{xi_bin[1]}" + r"]$",
        markersize=5,  # 6.5
    )


# plt.xscale('log')
# plt.yscale('log')
# plt.legend()

# plt.show()
plt.savefig("shear-closed-form.png", dpi=400)
