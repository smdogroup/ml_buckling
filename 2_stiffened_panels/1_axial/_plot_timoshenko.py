import pandas as pd
import matplotlib.pyplot as plt
import niceplots
import ml_buckling as mlb

"""
plotting script for previously saved data from the script
3_timoshenko_dataset.py
"""

if __name__=="__main__":

    df = pd.read_csv("axialCF-FEA.csv")
    X = df[["rho0", "gamma"]].to_numpy()
    Y = df[["eig_CF", "eig_FEA"]].to_numpy()
    
    rho0 = X[:,0]
    gamma = X[:,1]
    eig_CF = Y[:,0]
    eig_FEA = Y[:,1]

    gamma_vec = [0.0, 2.0, 5.0, 10.0]

    plt.style.use(niceplots.get_style())

    plt.figure("this")
    colors = mlb.six_colors2[:4][::-1]

    for igamma, gamma_val in enumerate(gamma_vec[::-1]):

        gamma_mask = gamma_val == gamma

        # plot the FEA
        # rho0_FEA = rho0[gamma_mask]
        # N11_FEA = eig_FEA[gamma_mask]
        # plt.plot(rho0_FEA, N11_FEA, "o", label=r"$\gamma = " + f"{gamma_val:.1f}" + r"$", color=colors[igamma], alpha=0.9)

        # plot the closed-form
        rho0_CF = rho0[gamma_mask]
        N11_CF = eig_CF[gamma_mask]
        plt.plot(rho0_CF, N11_CF, "-", label=None, color=colors[igamma], linewidth=2.5)

        # # plot the FEA
        rho0_FEA = rho0[gamma_mask]
        N11_FEA = eig_FEA[gamma_mask]
        # plot every other point for now when there were 100 per line (50 per line is a good density where can see points and line separately)
        plt.plot(rho0_FEA, N11_FEA, "o", label=r"$\gamma = " + f"{gamma_val:.1f}" + r"$", color=colors[igamma], alpha=0.8, markersize=5)
        # rho0_FEA[::2], N11_FEA[::2] (when there were 100 points and wanted only 50 of them)

    # finish making the plot
    plt.legend()
    # plt.title(r"$\gamma = 11.25$")
    plt.xlabel(r"$\rho_0$")
    plt.ylabel(r"$N_{11,cr}^*$")
    plt.xscale('log')
    plt.yscale('log')
    plt.margins(x=0.05, y=0.05)
    # plt.show()
    plt.savefig("axial-CF-vs-FEA.png", dpi=400)
    plt.savefig("axial-CF-vs-FEA.svg", dpi=400)
    