import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import niceplots

islender = 2
model_df = pd.read_csv(f"slender{islender}-model-fit_model.csv")
rho_0 = model_df['rho_0'].to_numpy()
mean = model_df['mean'].to_numpy()
std_dev = model_df['std_dev'].to_numpy() * 2

data_df = pd.read_csv(f"slender{islender}-model-fit_data.csv")
AR = data_df['AR'].to_numpy()
lam = data_df['lam'].to_numpy()

plt.style.use(niceplots.get_style())
plt.figure("check model")#, figsize=(8, 6))
plt.margins(x=0.05, y=0.05)
plt.title(f"b/h in [50,100]")
ax = plt.subplot(111)

colors = plt.cm.viridis(np.linspace(0, 1, 5))

ax.fill_between(
    x=rho_0,
    y1=mean - 3 * std_dev,
    y2=mean + 3 * std_dev,
    label='3-sigma',
    color=colors[0]
)
ax.fill_between(
    x=rho_0,
    y1=mean - std_dev,
    y2=mean + std_dev,
    label='1-sigma',
    color=colors[1]
)
ax.plot(
   rho_0, mean, "k", label="mean", linewidth=2
)  # , label=f"D*-[{Dstar_bin[0]},{Dstar_bin[1]}]""
ax.plot(
    AR, lam, "o", markersize=3, label="train-data",
    color=colors[2], markeredgecolor=colors[2]
)  # , label=f"D*-[{Dstar_bin[0]},{Dstar_bin[1]}]""

# outside of for loop save the plot
plt.xlabel(r"$\log(\rho_0)$")
plt.ylabel(r"$\log(\lambda_{min}^*)$")
plt.legend()
# plt.xlim(np.log(0.1), np.log(20.0))
# plt.ylim(np.log(2.0), np.log(20.0))
plt.savefig(f"slender{islender}-model-fit.png", dpi=400)