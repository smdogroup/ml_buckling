import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sample data structure (replace with your actual dataframe)
df = pd.DataFrame({
    "Kernel": ["Buckling+RQ", "Buckling+RQ", "Buckling+SE", "Buckling+SE",
               "SE", "SE", "Matern-3/2", "Matern-3/2",
               "Matern-5/2", "Matern-5/2", "RQ", "RQ"],
    "Type": ["Axial+Affine", "Shear+NoAffine", "Axial+Affine", "Shear+NoAffine",
             "Axial+Affine", "Shear+Affine", "Axial+Affine", "Shear+Affine",
             "Axial+Affine", "Shear+Affine", "Axial+Affine", "Shear+Affine"],
    "Value": np.random.rand(12)  # Replace with actual values
})

# Define the six categories with the special cases
selected_groups = [
    ("Buckling+RQ", "Axial+Affine"), ("Buckling+RQ", "Shear+NoAffine"),
    ("Buckling+SE", "Axial+Affine"), ("Buckling+SE", "Shear+NoAffine"),
    ("SE", "Axial+Affine"), ("SE", "Shear+Affine"),
    ("Matern-3/2", "Axial+Affine"), ("Matern-3/2", "Shear+Affine"),
    ("Matern-5/2", "Axial+Affine"), ("Matern-5/2", "Shear+Affine"),
    ("RQ", "Axial+Affine"), ("RQ", "Shear+Affine")
]

# Filter dataframe to only include the selected groups
df_subset = df[df.apply(lambda row: (row["Kernel"], row["Type"]) in selected_groups, axis=1)]

# Sort to ensure the order is correct
df_subset["SortKey"] = df_subset["Kernel"].map({
    "Buckling+RQ": 0, "Buckling+SE": 1, "SE": 2,
    "Matern-3/2": 3, "Matern-5/2": 4, "RQ": 5
}) + df_subset["Type"].map({"Axial+Affine": 0, "Shear+Affine": 1, "Shear+NoAffine": 1.5})
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
        df_plot["Value"] = np.log1p(df_plot["Value"])

    # Plot bars
    ax.bar(bar_positions - bar_width/2, df_plot[df_plot["Type"] == "Axial+Affine"]["Value"], bar_width, label="Axial+Affine")
    ax.bar(bar_positions + bar_width/2, df_plot[df_plot["Type"].isin(["Shear+Affine", "Shear+NoAffine"])]["Value"], bar_width, label="Shear Variant")

    # Labels and formatting
    ax.set_ylabel("Value (log)" if log_transform else "Value")
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(["Buckling+RQ", "Buckling+SE", "SE", "Matern-3/2", "Matern-5/2", "RQ"], rotation=45)
    ax.legend()

plt.tight_layout()
plt.show()
