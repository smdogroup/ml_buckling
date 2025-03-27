import matplotlib.pyplot as plt
from closed_form_dataset import nan_extrap_data
from plot_utils import plot_surface

def plot_GPs(
    X_plot,
    Y_plot_truth,
    Y_plot_pred,
    folder_name:str,
    base_name:str,
    axial:bool=True,
    affine:bool=True,
    log:bool=True,
    show:bool=False,
    file_ext:str="svg", # or "png"
    
):
    
    Y = Y_plot_pred
    Y_plot_pred_extrap = nan_extrap_data(X_plot, Y, log=log, affine=affine,
                                        nan_extrap=False)
    Y_plot_pred_interp = nan_extrap_data(X_plot, Y, log=log, affine=affine,
                                        nan_extrap=True)

    # plot with gamma
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    colors = ["blue", "gray"]
    for i,Yp in enumerate([Y_plot_pred_extrap, Y_plot_pred_interp]):
        plot_surface(X_plot, Y=Yp, 
            Y_truth=Y_plot_truth,
            # affine_shift=th[3] if args.kernel == 5 else None,
            surf_color_map=colors[i],
            var_exclude_ind=2, 
            nx1=20, nx2=10,
            var_exclude_range=[0.3]*2,
            ax=ax, show=False)

    ax.set_zlim(0.0, 6.0)

    fs = 14
    fw = 'bold'
    lp = 10

    load_str = "axial" if axial else "shear"
    affine_str = "affine" if affine else "not-affine"
    log_str = "log" if log else "not-log"
    plt.title(f"{load_str}, {affine_str}, {log_str}")

    ax.set_xlabel(r"$\mathbf{\ln(\rho_0^*)}$" if affine else r"$\mathbf{\ln(\rho_0)}$", fontsize=fs, fontweight=fw, labelpad=lp)
    ax.set_ylabel(r"$\mathbf{\ln(1+\gamma)}$", fontsize=fs, fontweight=fw, labelpad=lp)
    ax.set_zlabel(r"$\mathbf{\ln(\overline{N}_{11}^{cr})}$" if axial else r"$\mathbf{\ln(\overline{N}_{12}^{cr})}$", fontsize=fs, fontweight=fw, labelpad=6)

    if show:
        plt.show()
    else:
        axial_str = "axial" if axial else "shear"
        log_str = "log" if log else "nolog"
        affine_str = "affine" if affine else "noaffine"
        # plt.savefig(f"{folder_name}/{base_name}.png")
        plt.savefig(f"{folder_name}/{base_name}.{file_ext}", dpi=400)