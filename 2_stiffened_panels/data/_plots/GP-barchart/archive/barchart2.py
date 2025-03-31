import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data extracted from the table
data = {
    "Kernel": [
        "Buckling + SE", "Buckling + RQ", "SE", "RQ", "Matérn-3/2", "Matérn-5/2",
        "Buckling + SE", "Buckling + RQ", "Buckling + SE", "Buckling + RQ", 
        "SE", "RQ", "Matérn-3/2", "Matérn-5/2", "Buckling + SE", "Buckling + RQ"
    ],
    "Log": ["No"]*8 + ["Yes"]*8,
    "Affine": ["No"]*2 + ["Yes"]*6 + ["No"]*2 + ["Yes"]*6,
    "R2_Axial_Interp": [0.9728, 0.9792, 0.9980, 0.9920, 0.9957, 0.9970, 0.9883, 0.9893, 
                         0.9969, 0.9954, 0.9964, 0.9944, 0.9971, 0.9964, 0.9956, 0.9972],
    "R2_Axial_Extrap": [0.3367, 0.3351, 0.5082, -0.0580, -0.0723, np.nan, 0.5270, 0.5269, 
                         0.9557, 0.8931, 0.6355, 0.6511, 0.7771, 0.7709, 0.9450, 0.9966],
    "R2_Shear_Interp": [0.9488, 0.9649, 0.9925, 0.9839, 0.9960, 0.9969, 0.9176, 0.9312, 
                        0.9991, 0.9992, 0.9941, 0.9988, 0.9990, 0.9979, 0.9972, 0.9991],
    "R2_Shear_Extrap": [0.9499, 0.9510, np.nan, 0.0298, 0.1758, 0.0437, 0.8447, 0.8433, 
                        0.9990, 0.9964, 0.8627, 0.5706, 0.9197, 0.9771, 0.7043, 0.9541],
}

df = pd.DataFrame(data)

# Convert data for stacked bar chart
df_long = df.melt(id_vars=["Kernel", "Log", "Affine"], 
                  value_vars=["R2_Axial_Interp", "R2_Axial_Extrap", "R2_Shear_Interp", "R2_Shear_Extrap"],
                  var_name="Condition", value_name="R2")

# Remove NaN values
df_long = df_long.dropna()

# Split conditions into separate Axial/Shear + Interp/Extrap columns
df_long["Type"] = df_long["Condition"].apply(lambda x: "Axial" if "Axial" in x else "Shear")
df_long["Mode"] = df_long["Condition"].apply(lambda x: "Interp" if "Interp" in x else "Extrap")

# Pivot to prepare for stacking
df_pivot = df_long.pivot_table(index=["Kernel", "Type"], columns="Mode", values="R2").reset_index()

# Plot Axial and Shear separately
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for i, load_type in enumerate(["Axial", "Shear"]):
    subset = df_pivot[df_pivot["Type"] == load_type]
    
    # Bar positions
    bar_positions = np.arange(len(subset))
    
    # Plot stacked bars
    axes[i].bar(bar_positions, subset["Interp"], label="Interpolation", color="royalblue")
    axes[i].bar(bar_positions, subset["Extrap"], bottom=subset["Interp"], label="Extrapolation", color="orange")

    # Labels
    axes[i].set_xticks(bar_positions)
    axes[i].set_xticklabels(subset["Kernel"], rotation=45, ha="right")
    axes[i].set_ylabel("$R^2$")
    axes[i].set_title(f"$R^2$ for {load_type} Load")
    
    axes[i].legend()

plt.suptitle("Stacked Bar Chart of Gaussian Process Kernel Performance")
plt.tight_layout()
plt.show()
