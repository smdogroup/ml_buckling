"""
Compare ML and CF buckling loads implemented in TACS
@Author Sean Engelstad
@Date 08/28/2024

Use the repo https://github.com/smdogroup/ml_buckling
"""

import numpy as np
import matplotlib.pyplot as plt
import niceplots
from tacs import TACS, constitutive
import ml_buckling as mlb
import argparse

# parse the arguments
parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--load", type=str, default="Nx")
parent_parser.add_argument(
    "--show", default=True, action=argparse.BooleanOptionalAction
)
parent_parser.add_argument("--xi", type=float, default=1.0)
parent_parser.add_argument("--zeta", type=float, default=0.0)

args = parent_parser.parse_args()

DEG2RAD = np.pi / 180.0

dtype = TACS.dtype

# Create the orthotropic layup
ortho_prop = constitutive.MaterialProperties(
    rho=1550,
    specific_heat=921.096,
    E1=54e3,
    E2=18e3,
    nu12=0.25,
    G12=9e3,
    G13=9e3,
    G23=9e3,
    Xt=2410.0,
    Xc=1040.0,
    Yt=73.0,
    Yc=173.0,
    S12=71.0,
    alpha=24.0e-6,
    kappa=230.0,
)
ortho_ply = constitutive.OrthotropicPly(1e-3, ortho_prop)

# build the axial GP object (which is the main ML object we are testing for this example)
# however it is used inside of the constitutive object so we need to build that too
axialGP = constitutive.BucklingGP.from_csv(
    csv_file=mlb.axialGP_csv, theta_csv=mlb.axial_theta_csv
)
shearGP = constitutive.BucklingGP.from_csv(
    csv_file=mlb.shearGP_csv, theta_csv=mlb.shear_theta_csv
)
panelGP = constitutive.PanelGPs(axialGP=axialGP, shearGP=shearGP)

# don't put in any GP models (so using closed-form solutions rn)
ML_con = constitutive.GPBladeStiffenedShellConstitutive(
    panelPly=ortho_ply,
    stiffenerPly=ortho_ply,
    panelLength=2.0,
    stiffenerPitch=0.2,
    panelThick=1.5e-2,
    panelPlyAngles=np.array([0.0, 45.0, 90.0], dtype=dtype) * DEG2RAD,
    panelPlyFracs=np.array([0.5, 0.3, 0.2], dtype=dtype),
    stiffenerHeight=0.075,
    stiffenerThick=1e-2,
    stiffenerPlyAngles=np.array([0.0, 60.0], dtype=dtype) * DEG2RAD,
    stiffenerPlyFracs=np.array([0.6, 0.4], dtype=dtype),
    panelWidth=1.0,
    flangeFraction=0.8,
    panelGPs=panelGP,
)
# Set the KS weight really low so that all failure modes make a
# significant contribution to the failure function derivatives
# ML_con.setKSWeight(20.0)

# also build the CF constitutive
CF_con = constitutive.GPBladeStiffenedShellConstitutive(
    panelPly=ortho_ply,
    stiffenerPly=ortho_ply,
    panelLength=2.0,
    stiffenerPitch=0.2,
    panelThick=1.5e-2,
    panelPlyAngles=np.array([0.0, 45.0, 90.0], dtype=dtype) * DEG2RAD,
    panelPlyFracs=np.array([0.5, 0.3, 0.2], dtype=dtype),
    stiffenerHeight=0.075,
    stiffenerThick=1e-2,
    stiffenerPlyAngles=np.array([0.0, 60.0], dtype=dtype) * DEG2RAD,
    stiffenerPlyFracs=np.array([0.6, 0.4], dtype=dtype),
    panelWidth=1.0,
    flangeFraction=0.8,
)
# Set the KS weight really low so that all failure modes make a
# significant contribution to the failure function derivatives
# CF_con.setKSWeight(20.0)

# get the axial loads in nondimensional space w.r.t. rho_0
xi = args.xi
zeta = args.zeta
n = 500
plt.style.use(niceplots.get_style())
rho0_vec = np.linspace(0.2, 15.0, n)

colors = plt.cm.jet(np.linspace(0.0, 1.0, 10))
CF_vec = np.zeros((n,), dtype=TACS.dtype)
ML_vec = np.zeros((n,), dtype=TACS.dtype)

for igamma, gamma in enumerate([0.0, 1.0, 2.0, 5.0, 10.0]):
    # for igamma, gamma in enumerate([0.05, 0.64, 6.4, 53.0]):
    for i, rho0 in enumerate(rho0_vec):
        if args.load == "Nx":
            CF_vec[i] = CF_con.nondimCriticalGlobalAxialLoad(rho0, xi, gamma)
            ML_vec[i] = ML_con.nondimCriticalGlobalAxialLoad(rho0, xi, gamma, zeta)
        else:
            CF_vec[i] = CF_con.nondimCriticalGlobalShearLoad(rho0, xi, gamma)
            ML_vec[i] = ML_con.nondimCriticalGlobalShearLoad(rho0, xi, gamma, zeta)
    plt.plot(
        rho0_vec,
        CF_vec,
        "-",
        # label=f"CF-gam={gamma:.2f}",
        label=f"gam={gamma:.2f}",
        linewidth=2,
        color=colors[igamma],
    )
    plt.plot(
        rho0_vec,
        ML_vec,
        "--",
        # label=f"ML-gam={gamma:.2f}",
        color=colors[igamma],
    )

# plot it
plt.margins(x=0.05, y=0.05)
plt.xlabel(r"$\rho_0$")
if args.load == "Nx":
    plt.ylabel(r"$N_{11,cr}^*$")
else:
    plt.ylabel(r"$N_{12,cr}^*$")
plt.xscale("log")
plt.yscale("log")
plt.legend()

if args.show:
    plt.show()
else:
    plt.savefig("5-compare-shear.png", dpi=400)
