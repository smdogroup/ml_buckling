import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import tensorflow as tf

# define common plot bins
# -----------------------
xi_bins = [[0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1.05]]
rho0_bins = [[-2.5, -1.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 2.5]]
zeta_bins = [[0.0, 0.1], [0.1, 0.5], [0.5, 1.0], [1.0, 2.5]]
# gamma_bins = [[0.0, 0.1], [0.1, 1.0], [1.0, 2.0], [2.0, 3.0]]
gamma_vec = np.geomspace(1+0.0, 1+3.0, 6+1)-1
gamma_bins = [[gamma_vec[i], gamma_vec[i+1]] for i in range(6)]


# plotting methods
# ----------------

def plot_3d_gamma(
    X, Y,
    X_train,
    kernel_func,
    alpha_train,
    theta,
    xi_bin:list=[0.5, 0.7],
    zeta_bin:list=[0.2, 0.4],
    folder_name:str="",
    load_name:str="Nx", # or "Nxy"
    file_prefix:str="",
    AR_bounds:list=[0.2, 10.0],
    is_affine:bool=True,
    show:bool=False,
    save_npz:bool=False,
):

    # X is [ln(rho0 or rho0^*), ln(1+xi), ln(1+gamma), ln(1+10^3 * zeta)]
    # Y is ln(N11crbar) or ln(N12crbar)

    xi_mask = np.logical_and(xi_bin[0] <= X[:, 1], X[:, 1] <= xi_bin[1])
    avg_xi = 0.5 * (xi_bin[0] + xi_bin[1])

    zeta_mask = np.logical_and(zeta_bin[0] <= X[:, 3], X[:, 3] <= zeta_bin[1])
    avg_zeta = 0.5 * (zeta_bin[0] + zeta_bin[1])
    xi_zeta_mask = np.logical_and(xi_mask, zeta_mask)

    colors = ["#264653", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51"]
    colors = colors[::-1]

    plt.figure(f"3d rho_0, gamma, lam_star")
    ax = plt.axes(projection="3d", computed_zorder=False)

    # colors = plt.cm.jet(np.linspace(0.0, 1.0, len(gamma_bins)))

    for igamma, gamma_bin in enumerate(gamma_bins):

        gamma_mask = np.logical_and(
            gamma_bin[0] <= X[:, 2], X[:, 2] <= gamma_bin[1]
        )
        mask = np.logical_and(xi_zeta_mask, gamma_mask)

        rho0_bin = [np.log(AR_bounds[0]), np.log(AR_bounds[1])]
        rho0_mask = np.logical_and(
            rho0_bin[0] <= X[:,0], X[:,0] <= rho0_bin[1]
        )
        mask = np.logical_and(mask, rho0_mask)

        X_in_range = X[mask, :]
        Y_in_range = Y[mask, :]

        ax.scatter(
            X_in_range[:, 2], # ln(1+gamma)
            X_in_range[:, 0], # ln(rho0 or rho0^*)
            Y_in_range[:, 0],
            s=20,
            color=colors[igamma],
            edgecolors="black",
            zorder=2 + igamma,
        )

    # plot the scatter plot
    n_plot = 3000
    X_plot_mesh = np.zeros((30, 100))
    X_plot = np.zeros((n_plot, 4))
    ct = 0
    gamma_vec = np.linspace(0.0, 3.0, 30)
    AR_vec = np.log(np.linspace(AR_bounds[0], AR_bounds[1], 100))
    for igamma in range(30):
        for iAR in range(100):
            X_plot[ct, :] = np.array(
                [AR_vec[iAR], avg_xi, gamma_vec[igamma], avg_zeta]
                # [avg_xi, AR_vec[iAR], avg_zeta, gamma_vec[igamma]]
            )
            ct += 1

    # plot data model predictions
    x_plot_L = tf.expand_dims(X_plot, axis=1)
    x_train_R = tf.expand_dims(X_train, axis=0)
    Kplot = kernel_func(x_plot_L, x_train_R, theta)
    f_plot = Kplot @ alpha_train

    # make meshgrid of outputs
    GAMMA = np.zeros((30, 100))
    AR = np.zeros((30, 100))
    KMIN = np.zeros((30, 100))
    ct = 0
    for igamma in range(30):
        for iAR in range(100):
            GAMMA[igamma, iAR] = gamma_vec[igamma]
            AR[igamma, iAR] = AR_vec[iAR]
            KMIN[igamma, iAR] = f_plot[ct]
            ct += 1

    # plot the model curve
    # Creating plot
    face_colors = cm.Grays(0.4 * KMIN / KMIN)
    ax.plot_surface(
        GAMMA,
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
    ax.grid(False)

    # save data to a file (do later)
    if save_npz:
        np.savez(f"{folder_name}/{file_prefix}gamma-data.npz",
                    X=X, Y=Y,
                    GAMMA=GAMMA, AR=AR, KMIN=KMIN)

    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # ax.zaxis.pane.set_edgecolor('white')

    # save the figure
    fs1 = 16; ls1 = 10
    ax.set_xlabel(r"$\mathbf{\ln(1+\gamma)}$", fontsize=fs1, fontweight='bold', labelpad=ls1)
    ax.set_ylabel(r"$\mathbf{\ln(\rho_0^*)}$" if is_affine else r"$\mathbf{\ln(\rho_0)}$", fontsize=fs1, fontweight='bold', labelpad=ls1)
    if load_name == "Nx":
        ax.set_zlabel(r"$\mathbf{\ln(\overline{N}_{11}^{cr})}$", fontsize=fs1, fontweight='bold', labelpad=5)
    else:
        ax.set_zlabel(r"$\mathbf{\ln(\overline{N}_{12}^{cr})}$", fontsize=fs1, fontweight='bold', labelpad=5)
    ax.set_ylim3d(np.log(AR_bounds[0]), np.log(AR_bounds[1]))
    # ax.set_zlim3d(0.0, np.log(50.0))
    # ax.set_zlim3d(1.0, 3.0)
    ax.view_init(elev=20, azim=20, roll=0)
    plt.gca().invert_xaxis()

    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.zaxis.set_tick_params(labelsize=14)

    if show:
        plt.show()
    else:
        plt.savefig(f"{folder_name}/{file_prefix}_gamma-3d.png", dpi=400)
    plt.close(f"3d rho_0, gamma, lam_star")


def plot_3d_xi(
    X, Y,
    X_train,
    kernel_func,
    alpha_train,
    theta,
    gamma_bin:list=[0.0, 0.1],
    zeta_bin:list=[0.0, 1.0],
    folder_name:str="",
    load_name:str="Nx", # or "Nxy"
    file_prefix:str="",
    AR_bounds:list=[0.2, 10.0],
    is_affine:bool=True,
    show:bool=False,
    save_npz:bool=False,
):
    
    # X is [ln(rho0 or rho0^*), ln(1+xi), ln(1+gamma), ln(1+10^3 * zeta)]
    # Y is ln(N11crbar) or ln(N12crbar)
    
    # adjust bins for this plot in particular
    gamma_bins = [[0.0, 0.1], [0.1, 1.0], [1.0, 2.0], [2.0, 3.0]]
    xi_vec = np.geomspace(1+0.25, 1+1.05, 6+1)-1
    xi_bins = [[xi_vec[i], xi_vec[i+1]] for i in range(6)]

    # compute masks for this data
    gamma_mask = np.logical_and(gamma_bin[0] <= X[:, 2], X[:, 2] <= gamma_bin[1])
    avg_gamma = 0.5 * (gamma_bin[0] + gamma_bin[1])

    zeta_mask = np.logical_and(zeta_bin[0] <= X[:, 3], X[:, 3] <= zeta_bin[1])
    avg_zeta = 0.5 * (zeta_bin[0] + zeta_bin[1])
    gamma_zeta_mask = np.logical_and(gamma_mask, zeta_mask)

    colors = ["#264653", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51"]
    colors = colors[::-1]

    plt.figure(f"3d rho_0, xi, lam_star")
    ax = plt.axes(projection="3d", computed_zorder=False)

    # colors = plt.cm.jet(np.linspace(0.0, 1.0, len(gamma_bins)))

    for ixi, xi_bin in enumerate(xi_bins):

        xi_mask = np.logical_and(xi_bin[0] <= X[:, 1], X[:, 1] <= xi_bin[1])
        avg_xi = 0.5 * (xi_bin[0] + xi_bin[1])
        mask = np.logical_and(gamma_zeta_mask, xi_mask)

        rho0_bin = [np.log(AR_bounds[0]), np.log(AR_bounds[1])]
        rho0_mask = np.logical_and(
            rho0_bin[0] <= X[:,0], X[:,0] <= rho0_bin[1]
        )
        mask = np.logical_and(mask, rho0_mask)

        X_in_range = X[mask, :]
        Y_in_range = Y[mask, :]

        ax.scatter(
            X_in_range[:, 1], # ln(1+xi)
            X_in_range[:, 0], # ln(rho0 or rho0^*)
            Y_in_range[:, 0],
            s=20,
            color=colors[ixi],
            edgecolors="black",
            zorder=2 + ixi,
        )

    
    # plot the scatter plot
    n_plot = 3000
    # X_plot_mesh = np.zeros((30, 100))
    X_plot = np.zeros((n_plot, 4))
    ct = 0
    xi_vec = np.linspace(0.2, 1.0, 30)
    AR_vec = np.log(np.linspace(AR_bounds[0], AR_bounds[1], 100))
    for ixi in range(30):
        for iAR in range(100):
            X_plot[ct, :] = np.array(
                [AR_vec[iAR], xi_vec[ixi], avg_gamma, avg_zeta]
            )
            ct += 1

    # plot data model predictions
    x_plot_L = tf.expand_dims(X_plot, axis=1)
    x_train_R = tf.expand_dims(X_train, axis=0)
    Kplot = kernel_func(x_plot_L, x_train_R, theta)
    f_plot = Kplot @ alpha_train

    # make meshgrid of outputs
    XI = np.zeros((30, 100))
    AR = np.zeros((30, 100))
    KMIN = np.zeros((30, 100))
    ct = 0
    for ixi in range(30):
        for iAR in range(100):
            XI[ixi, iAR] = xi_vec[ixi]
            AR[ixi, iAR] = AR_vec[iAR]
            KMIN[ixi, iAR] = f_plot[ct]
            ct += 1

    # plot the model curve
    # Creating plot
    face_colors = cm.Grays(0.4 * KMIN / KMIN)
    ax.plot_surface(
        XI,
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
    ax.grid(False)

    # save data to a file
    if save_npz:
        np.savez(f"{folder_name}/{file_prefix}_xi-data.npz",
                    X=X, Y=Y,
                    XI=XI, AR=AR, KMIN=KMIN)

    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # ax.zaxis.pane.set_edgecolor('white')

    # save the figure
    fs1 = 16; ls1 = 10
    ax.set_xlabel(r"$\mathbf{\ln(1+\xi)}$", fontsize=fs1, fontweight='bold', labelpad=ls1)
    ax.set_ylabel(r"$\mathbf{\ln(\rho_0^*)}$" if is_affine else r"$\mathbf{\ln(\rho_0)}$", fontsize=fs1, fontweight='bold', labelpad=ls1)
    if load_name == "Nx":
        ax.set_zlabel(r"$\mathbf{\ln(\overline{N}_{11}^{cr})}$", fontsize=fs1, fontweight='bold', labelpad=5)
    else:
        ax.set_zlabel(r"$\mathbf{\ln(\overline{N}_{12}^{cr})}$", fontsize=fs1, fontweight='bold', labelpad=5)
    ax.set_ylim3d(np.log(AR_bounds[0]), np.log(AR_bounds[1]))
    # ax.set_zlim3d(0.0, np.log(50.0))
    # ax.set_zlim3d(1.0, 3.0)
    ax.view_init(elev=20, azim=20, roll=0)
    plt.gca().invert_xaxis()

    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.zaxis.set_tick_params(labelsize=14)

    if show:
        plt.show()
    else:
        plt.savefig(f"{folder_name}/{file_prefix}_xi-3d.png", dpi=400)
    plt.close(f"3d rho_0, xi, lam_star")