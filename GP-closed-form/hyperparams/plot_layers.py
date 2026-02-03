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
# Data (paste into script)
# -----------------------------
csv_text = """load,log,affine,nlayers,neurons,nparams,intR2,extR2
axial,yes,yes,1,4,21,0.4450,-0.4089
axial,yes,yes,1,16,81,0.8713,-0.1399
axial,yes,yes,1,64,321,0.9755,0.2385
axial,yes,yes,1,256,1281,0.9899,0.8030
axial,yes,yes,1,1024,5121,0.9915,0.9532
axial,yes,yes,1,4096,20481,0.9938,0.9727
axial,yes,yes,2,2,17,-0.8451,-0.2490
axial,yes,yes,2,4,41,-0.5219,-0.3465
axial,yes,yes,2,8,113,0.5545,-0.2599
axial,yes,yes,2,32,1217,0.9015,0.3345
axial,yes,yes,2,64,4481,0.9729,0.7009
axial,yes,yes,2,128,17153,0.9760,0.8060
axial,yes,yes,3,2,23,-0.7112,-0.2012
axial,yes,yes,3,4,61,-1.4488,-0.3194
axial,yes,yes,3,8,185,-0.068,-0.2378
axial,yes,yes,3,32,2273,0.7368,0.1957
axial,yes,yes,3,64,8641,0.8093,0.5733
axial,yes,yes,3,128,33665,0.8812,0.7293
axial,yes,yes,4,2,29,-0.5870,-0.2025
axial,yes,yes,4,4,81,-3.5607,-0.2730
axial,yes,yes,4,8,257,-1.084,-0.2519
axial,yes,yes,4,32,3329,0.3132,-0.0366
axial,yes,yes,4,64,12801,0.5693,0.1770
axial,yes,yes,4,128,50177,0.7855,0.5104
axial,no,no,1,4,21,-0.1872,-0.3909
axial,no,no,1,16,81,0.3927,-0.4843
axial,no,no,1,64,321,0.8591,-0.5950
axial,no,no,1,256,1281,0.9708,-0.6217
axial,no,no,1,1024,5121,0.9908,-0.0646
axial,no,no,1,4096,20481,0.9905,0.1837
axial,no,no,2,2,17,-0.8875,-0.3616
axial,no,no,2,4,41,-0.6030,-0.4491
axial,no,no,2,8,113,0.2179,-0.4774
axial,no,no,2,32,1217,0.9104,-0.4599
axial,no,no,2,64,4481,0.9637,-0.0873
axial,no,no,2,128,17153,0.9815,0.0630
axial,no,no,2,256,67073,0.9832,0.2665
axial,no,no,3,2,23,-0.3286,-0.1713
axial,no,no,3,4,61,-0.8405,-0.4203
axial,no,no,3,8,185,0.3072,-0.4318
axial,no,no,3,32,2273,0.8812,-0.1231
axial,no,no,3,64,8641,0.9377,0.0609
axial,no,no,3,128,33665,0.9669,-0.0051
axial,no,no,3,256,132865,0.9674,0.1593
axial,no,no,4,2,29,-0.3017,-0.1712
axial,no,no,4,4,81,-0.2018,-0.3732
axial,no,no,4,8,257,0.0805,-0.4146
axial,no,no,4,32,3329,0.8339,-0.3648
axial,no,no,4,64,12801,0.9019,-0.1928
axial,no,no,4,128,50177,0.9310,-0.3086
"""

df = pd.read_csv(pd.io.common.StringIO(csv_text))

# Numeric coercion
for c in ["nlayers", "neurons", "nparams", "intR2", "extR2"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# -----------------------------
# Plot helper
# -----------------------------
lw = 3
layer_color = {1: colors[0], 2: colors[1], 3: colors[2], 4: colors[3]}

def plot_panel(ax, subdf, title):
    subdf = subdf.sort_values(["nlayers", "nparams"])
    for nl in sorted(subdf["nlayers"].unique()):
        d = subdf[subdf["nlayers"] == nl].sort_values("nparams")
        ax.plot(
            d["nparams"], d["extR2"],
            marker="o",
            linestyle="-",
            color=layer_color.get(nl, "k"),
            markersize=ms,
            linewidth=lw,
            label=f"{nl} layer" + ("s" if nl != 1 else "")
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
plt.savefig("extrapR2_vs_nparams_axial_log_affine_vs_plain.png", dpi=300, bbox_inches="tight")
plt.savefig("ann_nlayers_extrap.svg", dpi=300, bbox_inches="tight")
plt.show()
