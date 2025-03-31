import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ml_buckling as mlb

# Data
# data = {
#     "Kernel": [
#         "Buckling + SE", "Buckling + RQ", "SE", "RQ", "Matérn-3/2", "Matérn-5/2",
#         "Buckling + SE", "Buckling + RQ", "Buckling + SE", "Buckling + RQ", 
#         "SE", "RQ", "Matérn-3/2", "Matérn-5/2", "Buckling + SE", "Buckling + RQ"
#     ],
#     "Log": ["No"]*8 + ["Yes"]*8,
#     "Affine": ["No"]*2 + ["Yes"]*6 + ["No"]*2 + ["Yes"]*6,
#     "R2_Axial_Interp": [0.9728, 0.9792, 0.9980, 0.9920, 0.9957, 0.9970, 0.9883, 0.9893, 
#                          0.9969, 0.9954, 0.9964, 0.9944, 0.9971, 0.9964, 0.9956, 0.9972],
#     "R2_Axial_Extrap": [0.3367, 0.3351, 0.5082, -0.0580, -0.0723, np.nan, 0.5270, 0.5269, 
#                          0.9557, 0.8931, 0.6355, 0.6511, 0.7771, 0.7709, 0.9450, 0.9966],
#     "R2_Shear_Interp": [0.9488, 0.9649, 0.9925, 0.9839, 0.9960, 0.9969, 0.9176, 0.9312, 
#                         0.9991, 0.9992, 0.9941, 0.9988, 0.9990, 0.9979, 0.9972, 0.9991],
#     "R2_Shear_Extrap": [0.9499, 0.9510, np.nan, 0.0298, 0.1758, 0.0437, 0.8447, 0.8433, 
#                         0.9990, 0.9964, 0.8627, 0.5706, 0.9197, 0.9771, 0.7043, 0.9541],
# }

colors = mlb.six_colors2[::-1]

data = {
    "Kernel" : [
        "Buckling+SE", # (Axial-Aff., Shear-NoAff.)
        "Buckling+RQ", # (Axial-Aff., Shear-NoAff.)
        "SE",
        "RQ",
        "Matérn-3/2",
        "Matérn-5/2",
    ]*2,
    "Log": ["No"]*6 + ["Yes"]*6,
    "R2_Axial_Interp": [0.9883, 0.9893, 0.9980, 0.9920, 0.9957, 0.9970,
                            0.9956, 0.9972, 0.9964, 0.9944, 0.9971, 0.9964],
    "R2_Axial_Extrap": [0.5270, 0.5269, 0.5082, -0.0580, -0.0723, np.nan,
                        0.9450, 0.9966, 0.6355, 0.6511, 0.7771, 0.7709],
    "R2_Shear_Interp": [0.9488, 0.9649, 0.9925, 0.9839, 0.9960, 0.9176,
                        0.9991, 0.9992, 0.9941, 0.9988, 0.9990, 0.9979],
    "R2_Shear_Extrap": [0.9499, 0.9510, np.nan, 0.0298, 0.1758, 0.0437,
                        0.9990, 0.9964, 0.8627, 0.5706, 0.9197, 0.9771],
}

df = pd.DataFrame(data)

# Convert to long format
df_long = df.melt(id_vars=["Kernel", "Log"], 
                  value_vars=["R2_Axial_Interp", "R2_Axial_Extrap", "R2_Shear_Interp", "R2_Shear_Extrap"],
                  var_name="Condition", value_name="R2").dropna()

# Extract Type & Mode
df_long["Type"] = df_long["Condition"].apply(lambda x: "Axial" if "Axial" in x else "Shear")
df_long["Mode"] = df_long["Condition"].apply(lambda x: "Interp" if "Interp" in x else "Extrap")

# Special rule for Buckling + SE & Buckling + RQ
# df_long["Custom_Affine"] = df_long.apply(
#     lambda row: "Yes" if row["Kernel"] not in ["Buckling + SE", "Buckling + RQ"]
#     else ("Yes" if row["Type"] == "Axial" else "No"), axis=1
# )

# Pivot data
df_pivot = df_long.pivot_table(index=["Kernel", "Log", "Type"], 
                               columns="Mode", values="R2").reset_index()

# # Unique x-axis labels
# df_pivot["Group"] = df_pivot["Kernel"] + "\nAffine: " + df_pivot["Custom_Affine"]

# Separate Log vs. No Log
df_no_log = df_pivot[df_pivot["Log"] == "No"].sort_values(by="Kernel") # by="Group"
df_log = df_pivot[df_pivot["Log"] == "Yes"].sort_values(by="Kernel")

# Fix bar positions
bar_positions_no_log = np.arange(len(df_no_log) // 2)
bar_positions_log = np.arange(len(df_log) // 2)

# Plot setup: Vertically stacked
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharey=True)

other_colors = ["royalblue", "orange", "lightblue", "gold"]

fs1 = 16
fw = 'bold'
ct = 0

for ax, df_subset, title, bar_positions in zip(axes, [df_no_log, df_log], ["No Log Transform", "Log Transform"], [bar_positions_no_log, bar_positions_log]):
    bar_width = 0.4
    ct += 1

    # Separate Axial and Shear
    axial = df_subset[df_subset["Type"] == "Axial"]
    shear = df_subset[df_subset["Type"] == "Shear"]

    # Plot Axial 
    ax.bar(bar_positions - bar_width/2, axial["Interp"], bar_width, 
           label="Axial Interp", color=colors[0])
    ax.bar(bar_positions - bar_width/2, axial["Extrap"], bar_width, 
           bottom=axial["Interp"], label="Axial Extrap", color=colors[2])

    # Plot Shear 
    ax.bar(bar_positions + bar_width/2, shear["Interp"], bar_width, 
           label="Shear Interp", color=colors[4])
    ax.bar(bar_positions + bar_width/2, shear["Extrap"], bar_width, 
           bottom=shear["Interp"], label="Shear Extrap", color=colors[5])

    # ax.set_xticks(bar_positions)
    ax.set_xticks(np.arange(len(df_subset["Kernel"].unique())))
    ax.set_xticklabels(df_subset["Kernel"].unique(), rotation=45, ha="right", fontsize=10, fontweight=fw)
    #ax.set_title(r"$\mathbf{R^2}$ Performance " + f"({title})", fontsize=14, fontweight=fw)
    ax.set_ylabel(r"Test $\mathbf{R^2}$", fontsize=fs1, fontweight=fw, labelpad=10)
    
#     ax.legend()
    # Move the legend outside to the right
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12, prop={'weight' : 'bold'})
    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.tick_params(axis='y', labelsize=10)
    #plt.setp(ax.get_yticklabels(), fontweight='bold')

    # add (a), (b) labels
    #mystr = "(a)" if ct == 1 else "(b)"
    #ax.text(0.02, 0.95, mystr, transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

fig.text(0.005, 0.99, "(a)", fontsize=18, fontweight='bold', ha='left', va='top', transform=fig.transFigure)
fig.text(0.005, 0.49, "(b)", fontsize=18, fontweight='bold', ha='left', va='top', transform=fig.transFigure)

plt.tight_layout()
# plt.show()
plt.savefig("GP_closedform_barchart.png", dpi=400)
plt.savefig("GP_closedform_barchart.svg", dpi=400)
