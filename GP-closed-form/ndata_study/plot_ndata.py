import pandas as pd
import matplotlib.pyplot as plt
import niceplots

# -----------------------------
# Style (same as your script)
# -----------------------------
plt.style.use(niceplots.get_style())

ms = 7
fs = 20
text_fs = 14

plt.rcParams.update({
    'font.size': fs,
    'axes.titlesize': fs,
    'axes.labelsize': fs,
    'xtick.labelsize': fs,
    'ytick.labelsize': fs,
    'legend.fontsize': fs,
    'figure.titlesize': fs,
})

colors = ["#d95a72", "#f3d27d", "#3cc7a1", "#3b90b3"]

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("ndata.csv")

for c in ["dataf", "ndata", "interp_R2", "extrap_R2"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# lw = 3.5
lw = 3

# -----------------------------
# Plot helper
# -----------------------------
def plot_load(ax, df_load, title):
    # Extend each curve flat to (max ndata + 1) using last available value
    target_max = int(df_load["ndata"].max())
    extend_x = target_max + 1

    styles = [
        ("GP", "interp_R2",      "-",  "s", colors[0], "GP-interp"),
        ("GP", "extrap_R2",      "--", "s", colors[1], "GP-extrap"),
        ("ANN_relu", "interp_R2","-",  "o", colors[2], "ANN–interp"),
        ("ANN_relu", "extrap_R2","--", "o", colors[3], "ANN–extrap"),
    ]

    for model, ycol, ls, mk, col, label in styles:
        sub = df_load[df_load["model"] == model][["ndata", ycol]].dropna()
        if len(sub) == 0:
            continue

        sub = sub.sort_values("ndata")
        last_x = float(sub["ndata"].iloc[-1])
        last_y = float(sub[ycol].iloc[-1])

        # extend (flat) to max+1
        if last_x < extend_x:
            sub = pd.concat(
                [sub, pd.DataFrame({"ndata": [extend_x], ycol: [last_y]})],
                ignore_index=True
            )

        ax.plot(
            sub["ndata"], sub[ycol],
            linestyle=ls,
            marker=mk,
            color=col,
            markersize=ms,
            linewidth=lw,
            label=label,
        )

    ax.set_xscale("log")
    ax.set_ylim(-1.0, 1.05)
    ax.set_xlabel("Number of data points")
    ax.set_ylabel(r"$R^2$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

# -----------------------------
# Figure
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

plot_load(axes[0], df[df["load"] == "axial"], "Axial")
plot_load(axes[1], df[df["load"] == "shear"], "Shear")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, -0.05)
)

import numpy as np
for ax in axes:
    ax.set_xlim(10, 2e4)
    # xticks = ax.get_xticks()
    # xlabels = [f"$10^{{{int(np.log10(x))}}}$" if x > 0 else "" for x in xticks]
    # ax.set_xticklabels(xlabels)


    ax.set_xticks([10, 1e2, 1e3, 1e4])
    ax.set_xticklabels([r"$10^{1}$", r"$10^{2}$", r"$10^{3}$", r"$10^{4}$"]) #, r"$2 \cdot 10^{4}$"])

plt.margins(x=0.05, y=0.05)

plt.tight_layout()
plt.savefig("R2_vs_ndata_axial_shear.png", dpi=300, bbox_inches="tight")
plt.savefig("R2_vs_ndata_axial_shear.svg", dpi=300, bbox_inches="tight")
plt.show()
