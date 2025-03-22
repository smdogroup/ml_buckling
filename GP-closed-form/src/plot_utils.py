import numpy as np
import matplotlib.pyplot as plt

def plot_surface(
    X, Y, # Y here is predictions
    nx1, nx2,
    Y_truth=None,
    var_exclude_ind=2, 
    var_exclude_range:list=[0.2, 0.4], 
    log_xi:bool=False,
    affine_shift=None,
    surf_color_map="gray", # "gray", "blue", etc.
    ax=None, show=True,
):
    
    from mpl_toolkits import mplot3d
    from matplotlib import cm

    Y_truth = Y if Y_truth is None else Y_truth
    
    # choose variable to exclude from plot, e.g. if var_exclude_ind = 2 then xi
    var_ind_list = [0,1,2]
    var_name_list = [r"$\rho_0", r"$\gamma", r"$\xi$"]
    del var_ind_list[var_exclude_ind]
    del var_name_list[var_exclude_ind]
    x1_ind = var_ind_list[0]
    x2_ind = var_ind_list[1]

    # now plot the dataset in (rho0, gamma, Nbar_cr) for xi slice say xi = 0.3
    exclude_mask = np.logical_and(
        var_exclude_range[0] <= X[:,var_exclude_ind],
        X[:,var_exclude_ind] <= var_exclude_range[1],
    )
    x1 = X[exclude_mask, x1_ind]
    x2 = X[exclude_mask,x2_ind]
    Ncr = Y[exclude_mask]

    # print(f"{x1_ind=} {x2_ind=}")

    # now turn into meshgrid dataset form
    my_shape = (nx1, nx2)
    X1 = x1.reshape(my_shape)
    X2 = x2.reshape(my_shape)
    NCR = Ncr.reshape(my_shape)

    # plt.figure(f"3d rho_0, gamma, lam_star")
    ax = plt.axes(projection="3d", computed_zorder=False) if ax is None else ax

    colors = ["#264653", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51"]

    if var_exclude_ind == 2: # exclude xi, so use gamma here
        x2_vec2 = np.geomspace(1.0, 16, 6+1) - 1.0
        bins = [[x2_vec2[i], x2_vec2[i+1]] for i in range(6)]
    else: # xi
        x2_vec2 = np.geomspace(0.3, 1.5, 5+1)
        bins = [[x2_vec2[i], x2_vec2[i+1]] for i in range(5)]
    # print(f"{bins=}")
    for igamma, bin in enumerate(bins):

        if np.max(X[:,0]) == 10.0: # linear scale
            this_gamma = X[:,1]
            this_xi = X[:,2]
        else: # log scale
            this_gamma = np.exp(X[:,1]) - 1.0
            if log_xi:
                this_xi = np.exp(X[:,2]) - 1.0
            else:
                this_xi = X[:,2]
            
        if var_exclude_ind == 2: # excludes xi so uses gamma
            gamma_mask = np.logical_and(
                bin[0] <= this_gamma, this_gamma <= bin[1]
            )
            xi_mask = np.logical_and(
                var_exclude_range[0] <= this_xi,
                this_xi <= var_exclude_range[1],
            )
        else: # uses xi bins
            gamma_mask = np.logical_and(
                var_exclude_range[0] <= this_gamma,
                this_gamma <= var_exclude_range[1],
            )
            xi_mask = np.logical_and(
                bin[0] <= this_xi, this_xi <= bin[1]
            )
        
        mask = np.logical_and(xi_mask, gamma_mask)

        X_in_range = X[mask, :]
        Y_in_range = Y_truth[mask]

        # print(f"{X_in_range.shape=}")

        if affine_shift is not None:
            X_in_range[:,0] -= affine_shift * X_in_range[:,1]

        ax.scatter(
            X_in_range[:, x1_ind], # rho0
            X_in_range[:, x2_ind], # gamma
            Y_in_range[:], # Ncr
            s=20,
            color=colors[igamma],
            edgecolors="black",
            zorder=2 + igamma,
        )

    # face_colors = cm.jet((KMIN - 0.8) / np.log(10.0))
    if surf_color_map == "gray":
        face_colors = cm.Grays(0.4 * X1 / X1)
    else:
        face_colors = cm.Blues(0.4 * X1 / X1)

    if affine_shift is not None:
        X1 -= affine_shift * X2

    ax.plot_surface(
        X1, X2, NCR,
        antialiased=False,
        facecolors=face_colors,
        alpha=0.5,
        zorder=1,
    )

    if show: plt.show()