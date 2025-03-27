import numpy as np, matplotlib.pyplot as plt
import pandas as pd
import niceplots

# need to regen data with 1_gen_mc_data.py, _solve_buckling = False and just printout ply_angle and xi data
df = pd.read_csv("data/Nxcrit_SS.csv")
arr = df.to_numpy()

slenderness = df['b/h'].to_numpy()
zeta = df['zeta'].to_numpy()

fs = 35
fs2 = 25
fw = 'bold'
lp = 6
ms = 40

# plt.figure(figsize=(8,6))
plt.scatter(slenderness, zeta, color='k', s=ms)


# plt.show()

plt.xlabel(r"Slenderness $b/t_f$", fontsize=fs, fontweight=fw, labelpad=lp)
plt.ylabel(r"$\mathbf{\zeta}$", fontsize=fs, fontweight=fw, labelpad=lp)
plt.xscale('log')
plt.yscale('log')

# Increase font size of tick labels
plt.tick_params(axis='both', which='major', labelsize=fs2)  # Modify font size for both x and y axis

# plt.xticks([0, 30, 45, 60, 90])

plt.tight_layout()  # Adjust layout to avoid clipping


#plt.show()
plt.savefig(f"plots/zeta-slender.png", dpi=400)
plt.close('all')