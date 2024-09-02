import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import niceplots

# TODO : need to redo this plotting script..

df = pd.read_csv("raw_data/Nx_stiffened.csv")

rho_0 = df["rho_0"].to_numpy()
xi = df["xi"].to_numpy()
gamma = df["gamma"].to_numpy()
zeta = df["zeta"].to_numpy()
lam_star = df["lambda_star"].to_numpy()
pred_type = df["pred_type"].to_numpy()

global_mask = pred_type == "global"
zeta_mask = zeta > 1300
base_mask = np.logical_and(global_mask, zeta_mask)

min_gamma = np.min(gamma[base_mask])
print(f"min gamma = {min_gamma}")
max_gamma = np.max(gamma[base_mask])
print(f"max gamma = {max_gamma}")

min_lam = np.min(lam_star[base_mask])
print(f"min_lam = {min_lam}")
max_lam = np.max(lam_star[base_mask])
print(f"max_lam = {max_lam}")


gamma_bins = [[np.power(10.0,i),np.power(10.0,i+1)] for i in range(-2,2,1)]
# print(f"gamma bins = {gamma_bins}")

plt.style.use(niceplots.get_style())
plt.figure("something")

colors = plt.cm.jet(np.linspace(0.0, 1.0, len(gamma_bins)))

for igamma,gamma_bin in enumerate(gamma_bins[::-1]):
    gamma_mask = np.logical_and(
        gamma_bin[0] <= gamma,
        gamma < gamma_bin[1]
    )

    mask = np.logical_and(base_mask, gamma_mask)
    
    plt.plot(rho_0[mask], lam_star[mask], "o", color=colors[igamma],
             label=r"$\gamma in [" + f"{gamma_bin[0]},{gamma_bin[1]}" + r"]$")
    
plt.xlabel(r"$\rho_0$")
plt.ylabel(r"$\lambda_{global}^*$")
# plt.show()
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.savefig("plots/stiffened_panel.png", dpi=400)