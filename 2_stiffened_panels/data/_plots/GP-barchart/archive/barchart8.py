import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Actual data as per your provided values
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

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Define the six categories with the special cases
selected_groups = [
    ("Buckling + RQ", "Axial+Affine"), ("Buckling + RQ", "Shear+NoAffine"),
    ("Buckling + SE", "Axial+Affine"), ("Buckling + SE", "Shear+NoAffine"),
    ("SE", "Axial+Affine"), ("SE", "Shear+Affine"),
    ("Matern-3/2", "Axial+Affine"), ("Matern-3/2", "Shear+Affine"),
    ("Matern-5/2", "Axial+Affine"), ("Matern-5/2", "Shear+Affine"),
    ("RQ", "Axial+Affine"), ("RQ", "Shear+Affine")
]

# Filter dataframe to only include the selected groups
df_subset = df[df.apply(lambda row: (row["Kernel"], row["Affine"]) in selected_groups, axis=1)]

# Sort the values to ensure correct bar positioning
df_subset["SortKey"] = df_subset["Kernel"].map({
    "Buckling + RQ": 0, "Buckling + SE": 1, "SE": 2,
    "Matern-3/2": 3, "Matern-5/2": 4, "RQ": 5
}) + df_subset["Affine"].map({"Axial+Affine": 0, "Shear+Affine": 1, "Shear+NoAffine": 1.5})
df_subset = df_subset.sort_values("SortKey")

# Define bar positions correctly for six groups (each with two bars side by side)
num_groups = 6
bar_width = 0.35
bar_positions = np.arange(num_groups)

# Split into log and no-log subsets
fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

for i, log_transform in enumerate([True, False]):
    ax = axs[i]
    df_plot = df_subset.copy()

    # Apply log transformation if needed
    if log_transform:
        df_plot["R2_Axial_Interp"] = np.log1p(df_plot["R2_Axial_Interp"])
        df_plot["R2_Axial_Extrap"] = np.log1p(df_plot["R2_Axial_Extrap"])
        df_plot["R2_Shear_Interp"] = np.log1p(df_plot["R2_Shear_Interp"])
        df_plot["R2_Shear_Extrap"] = np.log1p(df_plot["R2_Shear_Extrap"])

    # Plot bars for Axial and Shear R^2 values
    ax.bar(bar_positions - bar_width/2, df_plot[df_plot["Affine"] == "Axial+Affine"]["R2_Axial_Interp"], bar_width, label="Axial R² (Interp)")
    ax.bar(bar_positions + bar_width/2, df_plot[df_plot["Affine"] == "Shear+NoAffine"]["R2_Shear_Interp"], bar_width, label="Shear R² (Interp)")

    # Labels and formatting
    ax.set_ylabel("Log(Value)" if log_transform else "Value")
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(["Buckling + RQ", "Buckling + SE", "SE", "Matern-3/2", "Matern-5/2", "RQ"], rotation=45)
    ax.legend()

plt.tight_layout()
plt.show()
