import numpy as np, matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import niceplots

df = pd.read_csv("_data/Nxcrit_SS.csv")
arr = df.to_numpy()
lxi = arr[:,1]
lzeta = arr[:,3]

xi = np.exp(lxi) - 1.0
zeta = (np.exp(lzeta) - 1.0 ) / 1e3

print(f"{zeta=}")

# plt.hist(xi, bins=20, edgecolor='black')
#plt.hist(lzeta, bins=20, edgecolor='black')

mcase = 1
#mcase = 2

#plt.style.use(niceplots.get_style())
sns.set_style("whitegrid")

# Create a fancy histogram
#f, ax = plt.subplots(figsize=(8, 5))
#sns.despine(f)
#f mcase == 1:
#   arr = xi
#f mcase == 2:
#   arr = lzeta
color = None
color = "#3498DB"
kde_color = "#E74C3C"
#color = "royalblue"
#sns.histplot(arr, bins=30, kde=True, color=color, edgecolor="black", alpha=0.7) #, kde_kws={'color' : kde_color})

#sns.kdeplot(arr, color="#E74C3C", lw=2)  # KDE in red

# Create colormap gradient
# cmap = "coolwarm"
# cmap = 'CuBn'
cmap='RdYlBu'

colors = sns.color_palette(cmap, as_cmap=True)

# for ct, arr in enumerate([xi, lzeta]):
for ct, arr in enumerate([xi, zeta]):

    plt.figure(figsize=(8, 6))

    if ct == 0:
        n, bins, patches = plt.hist(arr, bins=20, density=False, edgecolor='black', alpha=0.9)
    else:
        bin_edges = np.logspace(np.log10(1e-4), np.log10(1e-1), num=20)
        n, bins, patches = plt.hist(arr, bins=bin_edges, density=False, log=False, # was log=True
                                     edgecolor='black', alpha=0.9)
        # bin_edges = np.logspace(np.log10(min(data)), np.log10(max(data)), num=30)
    
    # Apply gradient to bars
    for i, patch in enumerate(patches):
        patch.set_facecolor(colors(i / len(patches)))
    
    # Customize the look
    fs = 20
    fw = 'bold'
    lp = 6
    # plt.xlabel(r"$\boldsymbol{\xi}$" if ct == 0 else r"$\mathbf{\ln (1 + 10^3  \zeta)}$", fontsize=fs, fontweight=fw, labelpad=lp)
    plt.xlabel(r"$\boldsymbol{\xi}$" if ct == 0 else r"$\mathbf{\zeta}$", fontsize=fs, fontweight=fw, labelpad=lp)
    plt.ylabel("Frequency", fontsize=fs, fontweight=fw, labelpad=lp)
    #plt.title("Fancy Histogram with KDE", fontsize=14)
    # plt.grid(axis='y', linestyle="--", alpha=0.7)
    plt.grid(False)
    # plt.grid(False)

    if ct == 1:
        plt.xscale('log')

    # Increase font size of tick labels
    plt.tick_params(axis='both', which='major', labelsize=16)  # Modify font size for both x and y axis
    
    plt.tight_layout()  # Adjust layout to avoid clipping

    if ct == 1:
        plt.yticks([0, 100, 200, 300, 400])
        plt.margins(y=0.3)


    #plt.show()
    plt.savefig(f"plots/histo-{ct}.png", dpi=400)
    plt.close('all')

