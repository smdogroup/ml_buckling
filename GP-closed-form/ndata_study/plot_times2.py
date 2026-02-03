import pandas as pd
import matplotlib.pyplot as plt
import niceplots
import numpy as np

# -----------------------------
# Style (same as your script)
# -----------------------------
plt.style.use(niceplots.get_style())

ms = 8
# ms = 9
# ms = 7
fs = 20
text_fs = 14

plt.rcParams.update({
    'font.size': fs,
    'axes.titlesize': fs,
    'axes.labelsize': fs,
    'xtick.labelsize': fs,
    'ytick.labelsize': fs,
    'legend.fontsize': fs,   # keep global, override per-legend below
    'figure.titlesize': fs,
})

four_colors = ["#d95a72", "#f3d27d", "#3cc7a1", "#3b90b3"]
colors = ["#d95a72", "#3cc7a1", "#3b90b3"]

# colors = ["#3cc7a1", "#f3d27d", "#3b90b3"]

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("ndata2.csv")

for c in ["dataf", "ndata", "interp_R2", "extrap_R2", "train_time_sec"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Axial only
df = df[df["load"] == "axial"].copy()

# -----------------------------
# Build runtime components
# -----------------------------
buckling_hours_per_point = 0.011975  # hours per data point

df["data_gen_hr"] = buckling_hours_per_point * df["ndata"]
df["train_hr"] = df["train_time_sec"] / 3600.0
df["total_hr"] = df["data_gen_hr"] + df["train_hr"]

df["rel_extrap"] = 1.0 - df["extrap_R2"]
df["rel_extrap"] = df["rel_extrap"].clip(lower=0.0)

df_gp = df[df["model"] == "GP"].copy()
df_ann = df[df["model"] == "ANN_log"].copy()
df_ann_nlog = df[df["model"] == "ANN_nolog"].copy()

lw = 3.5

# -----------------------------
# Plot helpers
# -----------------------------
def plot_runtime_three_curves(ax, df_gp, df_ann):
    n_all = np.unique(
        np.concatenate([
            df_gp["ndata"].dropna().to_numpy(dtype=float),
            df_ann["ndata"].dropna().to_numpy(dtype=float)
        ])
    )
    n_all.sort()

    data_gen = buckling_hours_per_point * n_all

    gp_map = dict(zip(df_gp["ndata"].to_numpy(dtype=float), df_gp["train_hr"].to_numpy(dtype=float)))
    ann_map = dict(zip(df_ann["ndata"].to_numpy(dtype=float), df_ann["train_hr"].to_numpy(dtype=float)))

    gp_train = np.array([gp_map.get(n, np.nan) for n in n_all], dtype=float)
    ann_train = np.array([ann_map.get(n, np.nan) for n in n_all], dtype=float)

    ax.plot(n_all, data_gen,
            linestyle="-", marker=None, color=colors[1],
            linewidth=lw, label="data-gen")

    ax.plot(n_all, gp_train,
            linestyle="-", marker="s", color=colors[0],
            markersize=ms, linewidth=lw, label="train-GP")

    ax.plot(n_all, ann_train,
            linestyle="-", marker="o", color=colors[2],
            markersize=ms, linewidth=lw, label="train-ANN")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of data points")
    ax.set_ylabel("Runtime (hours)")
    ax.grid(True, alpha=0.3)

def plot_reliability_vs_runtime(ax, df_ann, df_gp):
    sub_ann2 = df_ann_nlog[["total_hr", "rel_extrap"]].dropna().sort_values("total_hr")
    sub_ann = df_ann[["total_hr", "rel_extrap"]].dropna().sort_values("total_hr")
    sub_gp  = df_gp[["total_hr", "rel_extrap"]].dropna().sort_values("total_hr")

    ax.plot(sub_gp["total_hr"], sub_gp["rel_extrap"],
            linestyle="-", marker="s", color=colors[0],
            markersize=ms, linewidth=lw, label="GP")

    ax.plot(sub_ann["total_hr"], sub_ann["rel_extrap"],
            linestyle="-", marker="o", color=colors[1],
            markersize=ms, linewidth=lw, label="ANN-L")
    
    ax.plot(sub_ann2["total_hr"], sub_ann2["rel_extrap"],
            linestyle="-", marker="o", color=colors[2],
            markersize=ms, linewidth=lw, label="ANN-NL")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total runtime (hours)")
    ax.set_ylabel(r"reliability: $1 - R^2_{\mathrm{extrap}}$")
    ax.grid(True, alpha=0.3)

# -----------------------------
# Figure (axial only)
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# (a) Runtime breakdown
plot_runtime_three_curves(axes[0], df_gp, df_ann)
axes[0].set_xlim(10, 2e4)
axes[0].set_xticks([10, 1e2, 1e3, 1e4])
axes[0].set_xticklabels([r"$10^{1}$", r"$10^{2}$", r"$10^{3}$", r"$10^{4}$"])

# Smaller legend + minimal padding so it doesn't force extra layout space
# axes[0].legend(
#     frameon=False,
#     loc="lower left",
#     bbox_to_anchor=(-0.10, 1.01),
#     ncol=3,
#     prop={"size": fs - 5},
#     columnspacing=0.6,
#     handletextpad=0.4,
#     handlelength=1.4,
#     borderaxespad=0.0,
#     labelspacing=0.2,
# )

# (b) Reliability vs runtime
plot_reliability_vs_runtime(axes[1], df_ann, df_gp)
# axes[1].legend(
#     frameon=False,
#     loc="lower left",
#     bbox_to_anchor=(0.18, 1.01),
#     ncol=2,
#     prop={"size": fs - 5},
#     columnspacing=0.8,
#     handletextpad=0.4,
#     handlelength=1.4,
#     borderaxespad=0.0,
#     labelspacing=0.2,
# )

# --- legends ABOVE the axes (outside), without shrinking font ---
axes[0].legend(
    frameon=False,
    loc="lower left",            # key: anchor legend's bottom edge
    bbox_to_anchor=(-0.05, 1.02),
    ncol=3,
    columnspacing=0.6,
    handletextpad=0.5,
    borderaxespad=0.0,
)

axes[1].legend(
    frameon=False,
    loc="lower left",            # key
    bbox_to_anchor=(-0.0, 1.02),
    # ncol=2,
    ncol=3,
    columnspacing=0.8,
    handletextpad=0.5,
    borderaxespad=0.0,
)

# axes[1].legend(
#     frameon=False,
#     loc="lower left",
#     bbox_to_anchor=(0.20, 1.02),
#     ncol=2,                    # 2 cols -> wraps to 2 rows for 3 items
#     columnspacing=0.6,         # tighter
#     handletextpad=0.35,        # tighter
#     handlelength=1.0,          # shorter line/handle
#     borderaxespad=0.0,
#     labelspacing=0.25,         # tighter vertical spacing between rows
# )


# Panel labels OUTSIDE axes (above-left)
axes[0].text(-0.16, 1.045, "(a)", transform=axes[0].transAxes,
             ha="left", va="bottom")
axes[1].text(-0.16, 1.045, "(b)", transform=axes[1].transAxes,
             ha="left", va="bottom")

plt.margins(x=0.05, y=0.05)

# IMPORTANT: use subplots_adjust instead of tight_layout to avoid wasted inter-axes space
fig.subplots_adjust(left=0.08, right=0.99, bottom=0.17, top=0.88, wspace=0.22)

plt.savefig("axial_runtime_and_reliability.png", dpi=300, bbox_inches="tight")
plt.savefig("axial_runtime_and_reliability.svg", dpi=300, bbox_inches="tight")
plt.show()
