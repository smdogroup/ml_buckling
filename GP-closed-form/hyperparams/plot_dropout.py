import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Optional style helper (same as your reference)
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
# Data: either read a file OR paste CSV text
# -----------------------------
# Option A (recommended): read your running CSV
df = pd.read_csv("../ann_results.csv")

# # Option B: paste here
# csv_text = """load,log,affine,dropout,nlayers,neurons,nparams,intR2,extR2
# axial,yes,yes,0.0,1,4,21,0.6107424543372568,-0.42150906514155895
# axial,yes,yes,0.0,1,16,81,0.9621390814987574,0.36425293236218537
# axial,yes,yes,0.0,1,64,321,0.9843610834334285,0.5539970612861101
# axial,yes,yes,0.0,1,256,1281,0.9937165637287821,0.8962369118498752
# axial,yes,yes,0.0,1,1024,5121,0.996422094149985,0.9497115648110941
# axial,yes,yes,0.0,1,4096,20481,0.9960890203587712,0.9805324300163968
# """
# df = pd.read_csv(pd.io.common.StringIO(csv_text))

# Numeric coercion
for c in ["dropout", "nlayers", "neurons", "nparams", "intR2", "extR2"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# -----------------------------
# Plot helper
# -----------------------------
lw = 3

# Map dropout -> color (assumes you use 0.0,0.1,0.2,0.3; otherwise it cycles)
dropouts_all = sorted([d for d in df["dropout"].dropna().unique()])
dropout_color = {d: colors[i % len(colors)] for i, d in enumerate(dropouts_all)}

def plot_panel(ax, subdf, title):
    # one curve per dropout
    subdf = subdf.sort_values(["dropout", "nparams"])

    for do in sorted(subdf["dropout"].dropna().unique()):
        d = subdf[subdf["dropout"] == do].sort_values("nparams")
        if len(d) == 0:
            continue

        ax.plot(
            d["nparams"], d["extR2"],
            marker="o",
            linestyle="-",
            color=dropout_color.get(do, "k"),
            markersize=ms,
            linewidth=lw,
            label=fr"dropout={do:.1f}",
        )

    ax.set_xscale("log")
    ax.set_xlim(1e1, 2e5)
    ax.set_ylim(-1.0, 1.05)

    ax.set_xlabel("Number of parameters")
    ax.set_ylabel(r"Extrapolation $R^2$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    ax.set_xticks([1e1, 1e2, 1e3, 1e4, 1e5])
    ax.set_xticklabels([r"$10^{1}$", r"$10^{2}$", r"$10^{3}$", r"$10^{4}$", r"$10^{5}$"])

# -----------------------------
# Filter subsets
# -----------------------------
df_a = df[(df["load"] == "axial") & (df["log"] == "yes") & (df["affine"] == "yes")]
df_b = df[(df["load"] == "axial") & (df["log"] == "no") & (df["affine"] == "no")]

# -----------------------------
# Figure
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

plot_panel(axes[0], df_a, "(a) Axial (log + affine)")
plot_panel(axes[1], df_b, "(b) Axial (no log, no affine)")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, -0.05),
)

plt.tight_layout()
plt.savefig("extrapR2_vs_nparams_axial_dropout_sweep.png", dpi=300, bbox_inches="tight")
plt.savefig("extrapR2_vs_nparams_axial_dropout_sweep.svg", dpi=300, bbox_inches="tight")
plt.show()
