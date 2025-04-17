import numpy as np
import matplotlib.pyplot as plt
import niceplots

import sys, pickle, os
from mpi4py import MPI
from funtofem import *

# get argparse
# parser = get_argparser()
# args = parser.parse_args()

comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))

# setup the model and solver
f2f_model = FUNtoFEMmodel("model")


# bodies and struct DVs
# ---------------------

wing = Body.aeroelastic("wing", boundary=2)

# make the component groups list for TACS
nribs = 23
nOML = nribs - 1
component_groups = [f"rib{irib}" for irib in range(1, nribs + 1)]
for prefix in ["spLE", "spTE", "uOML", "lOML"]:
    component_groups += [f"{prefix}{iOML}" for iOML in range(1, nOML + 1)]
component_groups = sorted(component_groups)

# make each struct design variable
for icomp, comp in enumerate(component_groups):
    Variable.structural(
        f"{comp}-" + TacsSteadyInterface.LENGTH_VAR, value=0.04
    ).set_bounds(
        lower=0.0,
        scale=1.0,
        state=True,  # need the length & width to be state variables
    ).register_to(
        wing
    )

    struct_active = True

    # stiffener pitch variable
    Variable.structural(f"{comp}-spitch", value=0.20).set_bounds(
        lower=0.05, upper=0.5, scale=1.0, active=struct_active
    ).register_to(wing)

    # panel thickness variable, shortened DV name for ESP/CAPS, nastran requirement here
    Variable.structural(f"{comp}-T", value=0.02).set_bounds(
        lower=0.002, upper=0.1, scale=100.0, active=struct_active
    ).register_to(wing)

    # stiffener height
    Variable.structural(f"{comp}-sheight", value=0.05).set_bounds(
        lower=0.0254 / 0.8, upper=0.1, scale=10.0, active=struct_active
    ).register_to(wing)

    # stiffener thickness
    Variable.structural(f"{comp}-sthick", value=0.02).set_bounds(
        lower=0.002, upper=0.1, scale=100.0, active=struct_active
    ).register_to(wing)

    Variable.structural(
        f"{comp}-" + TacsSteadyInterface.WIDTH_VAR, value=0.02
    ).set_bounds(
        lower=0.0,
        scale=1.0,
        state=True,  # need the length & width to be state variables
    ).register_to(
        wing
    )

wing.register_to(f2f_model)

# get ML design variables in a dictionary
f2f_model.read_design_variables_file(comm, "ML-sizing_design.txt")

ML_var_dict = {}
for var in f2f_model.get_variables():
    if var.analysis_type == "structural":
        ML_var_dict[var.name] = var.value.real

# get ML design variables in a dictionary
f2f_model.read_design_variables_file(comm, "CF-sizing_design.txt")

CF_var_dict = {}
for var in f2f_model.get_variables():
    if var.analysis_type == "structural":
        CF_var_dict[var.name] = var.value.real

# plot step function comparison of the upper and lower skin thicknesses
# ---------------------------------------------------------------------

plt.style.use(niceplots.get_style())
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# fig, ax = plt.subplots(2, 1, figsize=(15,10))
fig, ax = plt.subplots(1, 1, figsize=(7, 10))
for i, prefix in enumerate(["u", "l"]):
    for j, mydict in enumerate([ML_var_dict, CF_var_dict]):

        positions = []
        thicknesses = []

        N = 23
        for iOML in range(1, N-1 + 1):
            pthick = mydict[f"{prefix}OML{iOML}-T"]
            sthick = mydict[f"{prefix}OML{iOML}-sthick"]
            sheight = mydict[f"{prefix}OML{iOML}-sheight"]
            spitch = mydict[f"{prefix}OML{iOML}-spitch"]
            stiff_area = sthick * sheight * 1.8
            ethick = pthick + stiff_area / spitch
            ethick *= 1e3 # to mm

            # left side
            positions += [(iOML-1) / N, (iOML)/N]
            thicknesses += [ethick, ethick]

        middle_str = "upper" if prefix == "u" else "lower"
        ML_prefix = "ML" if j == 0 else "CF"
        ax[0].plot(positions, thicknesses, "--" if j == 1 else "-", color=colors[i], label=f"{ML_prefix}-{middle_str}-skin")
ax[0].legend()
# plt.show()
ax[0].set_ylabel("Effective Thickness (mm)", fontweight='bold')
xtick_positions = list(np.array([0.0, 3.0, 23.0])/23.0)  # The locations where you want the labels
xtick_labels = ['Root', 'SOB', 'Tip']  # The custom labels
ax[0].set_xticks(xtick_positions, xtick_labels)
for label in ax[0].get_yticklabels():
    label.set_fontweight('bold')
ax[0].margins(y=0.05)
ax[0].tick_params(axis='both', labelsize=18)
# plt.savefig("AOB-skins.png")

# plot step function comparison of the LE and TE spars
# ---------------------------------------------------------------------

# plt.figure(figsize=(15,5))
for i, prefix in enumerate(["LE", "TE"]):
    for j, mydict in enumerate([ML_var_dict, CF_var_dict]):

        positions = []
        thicknesses = []

        N = 23
        for iOML in range(1, N-1 + 1):
            pthick = mydict[f"sp{prefix}{iOML}-T"]
            sthick = mydict[f"sp{prefix}{iOML}-sthick"]
            sheight = mydict[f"sp{prefix}{iOML}-sheight"]
            spitch = mydict[f"sp{prefix}{iOML}-spitch"]
            stiff_area = sthick * sheight * 1.8
            ethick = pthick + stiff_area / spitch
            ethick *= 1e3 # to mm

            # left side
            positions += [(iOML-1) / N, (iOML)/N]
            thicknesses += [ethick, ethick]

        # middle_str = "LE" if prefix == "u" else "lower"
        ML_prefix = "ML" if j == 0 else "CF"
        ax[0].plot(positions, thicknesses, "--" if j == 1 else "-", color=colors[i], label=f"{ML_prefix}-{prefix}spar")
# ax[1].legend()
# ax[1].set_ylabel("Effective Thickness (mm)", fontweight='bold')
# xtick_positions = list(np.array([0.0, 3.0, 23.0])/23.0)  # The locations where you want the labels
# xtick_labels = ['Root', 'SOB', 'Tip']  # The custom labels
# ax[1].set_xticks(xtick_positions, xtick_labels)
# for label in ax[1].get_yticklabels():
#     label.set_fontweight('bold')
# ax[1].margins(y=0.05)
# ax[1].tick_params(axis='both', labelsize=18)
plt.savefig("AOB-thicknesses.png", dpi=400)
plt.savefig("AOB-thicknesses.svg", dpi=400)