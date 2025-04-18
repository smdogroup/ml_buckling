"""
check the implemented or saved kernels trained here in python
match the kernels implemented in TACS
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import niceplots, scipy, time, os
import argparse
from mpl_toolkits import mplot3d
from matplotlib import cm
import shutil, random, sys
# from archive._saved_kernel import kernel, axial_theta_opt, shear_theta_opt
from tacs import TACS, constitutive
import ml_buckling as mlb

sys.path.append("src/")
from kernel_library import *

# random seed (fixed for now..)
np.random.seed(123456)

# parse the arguments
parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--load", type=str, default="Nx")
# parent_parser.add_argument('--plotraw', default=False, action=argparse.BooleanOptionalAction)

args = parent_parser.parse_args()

assert args.load in ["Nx", "Nxy"]


kernel = buckling_RQ_kernel

load = args.load

# PRELIM LOAD THE DATASET, etc.
# -------------------------------------------------------------

# get the previous training weights from MLB package
if args.load == "Nx":
    csv_file = mlb.axialGP_csv
    theta_csv = mlb.axial_theta_csv
elif args.load == "Nxy":
    csv_file = mlb.shearGP_csv
    theta_csv = mlb.shear_theta_csv


import pandas as pd

dtype = TACS.dtype
df = pd.read_csv(csv_file)
Xtrain_mat = df[df.columns[1:5]].to_numpy(dtype=dtype)
alpha = df[df.columns[-1]].to_numpy(dtype=dtype)
alpha = np.reshape(alpha, (alpha.shape[0], 1))

theta_df = pd.read_csv(theta_csv)
theta_opt = theta_df[["theta"]].to_numpy()[:,0]

# flip the indices of Xtrain matrix
# otherwise gamma, zeta values are flipped
Xtrain_mat_tacs = Xtrain_mat[:, [0, 1, 3, 2]]
n_train = alpha.shape[0]
# no longer need to do this as fixed the order

# now is order [log(rho_0), log(1+xi), log(1+gamma), log(1+10^3 * zeta)]
X_train_mlb = Xtrain_mat[:, np.array([1,0,2,3])]

# affine transform
if args.load == "Nx":
    X_train_mlb[:,0] -= X_train_mlb[:,2] * 0.25

# print(f"{X_train_mlb=}")
# print(f"{alpha=}")
# exit()

# build the TACS GP model
# -------------------------------------------
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
axialGP = constitutive.BucklingGP.from_axial_csv(
    csv_file=mlb.axialGP_csv, theta_csv=mlb.axial_theta_csv
)
shearGP = constitutive.BucklingGP.from_shear_csv(
    csv_file=mlb.shearGP_csv, theta_csv=mlb.shear_theta_csv
)
panelGP = constitutive.PanelGPs(axialGP=axialGP, shearGP=shearGP)

# don't put in any GP models (so using closed-form solutions rn)
con = constitutive.GPBladeStiffenedShellConstitutive(
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
# con.setKS(10.0)

# COMPARE the kernel functions first
# -------------------------------------------
# ln(rho0), ln(1+xi), ln(1+gamma), ln(1+10^3*zeta)
c_Xtrain = np.random.rand(4).astype(TACS.dtype)
c_Xtest = np.random.rand(4).astype(TACS.dtype)

mlb_c_Xtrain = np.reshape(c_Xtrain.copy(), (1,1,4))
mlb_c_Xtest = np.reshape(c_Xtest.copy(), (1,1,4))

# affine transform
if args.load == "Nx":
    mlb_c_Xtrain[:,:,0] -= mlb_c_Xtrain[:,:,2] * 0.25
    mlb_c_Xtest[:,:,0] -= mlb_c_Xtest[:,:,2] * 0.25

# reorder inputs to TACS
# ln(1+xi), ln(rho0), ln(1+gamma), ln(1+10^3*zeta)
c_Xtrain = c_Xtrain[np.array([1,0,2,3])]
c_Xtest = c_Xtest[np.array([1,0,2,3])]
print(f"{c_Xtrain=} {c_Xtest=}")

mlb_kernel_res = kernel(mlb_c_Xtrain, mlb_c_Xtest, theta_opt)
tacs_kernel_res = axialGP.kernel(c_Xtrain, c_Xtest)

# useful check on whether the covariance for same inputs match
print(f"{mlb_kernel_res=}")
print(f"{tacs_kernel_res=}")
# exit()

# now also compare it again for prescribed xi, rho_0, gamma, zeta
xi = 1.0
rho_0 = 1.0
gamma = 0.0
zeta = 0.0

# predictions for MLB model
c_Xtest_left = np.array(
    [np.log(rho_0), np.log(1.0 + xi), 
     np.log(1.0 + gamma), np.log(1.0 + 1000.0 * zeta)]
).reshape((1,1,4)) # no affine transform needed because gamma = 0.0
x_train_right = tf.expand_dims(X_train_mlb, axis=0)
K_cross = kernel(c_Xtest_left, x_train_right, theta_opt)
print(f"{np.array(K_cross[:10])=}")
print(f"{alpha[:10]=}")
pred_log_load = np.dot(K_cross, alpha)
mlb_buckling_load = np.exp(pred_log_load)

# tacs prediction for single buckling load
if args.load == "Nx":
    tacs_buckling_load = con.nondimCriticalGlobalAxialLoad(rho_0, xi, gamma, zeta)
else:
    tacs_buckling_load = con.nondimCriticalGlobalShearLoad(rho_0, xi, gamma, zeta)

print(f"{mlb_buckling_load=}")
print(f"{tacs_buckling_load=}")

exit()


# PREDICTIONS for the ML_buckling model
# -----------------------------------------------------------

# predict and report the relative error on the test dataset
K_test_cross = np.array(
    [
        [kernel(Xtrain_mat[i, :], X_test[j, :], theta_opt) for i in range(n_train)]
        for j in range(n_test)
    ]
)
Y_test_pred_mlb_log = K_test_cross @ alpha
Y_test_pred_mlb = np.exp(Y_test_pred_mlb_log)

# PREDICTIONS for the TACS model
# -----------------------------------------------------------

if args.load == "Nx":
    Y_test_pred_tacs = np.array(
        [
            con.nondimCriticalGlobalAxialLoad(
                rho_0=np.exp(X_test[itest, 1]),
                xi=np.exp(X_test[itest, 0]) - 1.0,
                gamma=np.exp(X_test[itest, 3]) - 1,
                zeta=(np.exp(X_test[itest, 2]) - 1.0) / 1000,
            )
            for itest in range(X_test.shape[0])
        ]
    )
else:
    Y_test_pred_tacs = np.array(
        [
            con.nondimCriticalGlobalShearLoad(
                rho_0=np.exp(X_test[itest, 1]),
                xi=np.exp(X_test[itest, 0]) - 1.0,
                gamma=np.exp(X_test[itest, 3]) - 1,
                zeta=(np.exp(X_test[itest, 2]) - 1.0) / 1000,
            )
            for itest in range(X_test.shape[0])
        ]
    )


# COMPARE THE PREDICTIONS (should match well)
# -------------------------------------------

print(f"{Y_test_pred_mlb=}")
print(f"{Y_test_pred_tacs=}")

abs_err = np.abs(Y_test_pred_mlb - Y_test_pred_tacs)
rel_err = abs_err / Y_test_pred_mlb
max_rel_err = np.max(rel_err)
print(f"{max_rel_err=}")

# plot the comparison
import matplotlib.pyplot as plt

# import niceplots

# two parameters chosen as constant for plot comparison
xi = 1.0  # 0.4
zeta = 0.0

# get the axial loads in nondimensional space w.r.t. rho_0
n = 100
# plt.style.use(niceplots.get_style())
rho0_vec = np.linspace(0.5, 10.0, n)
N11cr_mlb = np.zeros((n,), dtype=dtype)
N11cr_tacs = np.zeros((n,), dtype=dtype)

colors = plt.cm.jet(np.linspace(0.0, 1.0, 8))

# for igamma,gamma in enumerate([0.0, 0.1, 0.5, 1.0]):
for igamma, gamma in enumerate([0.05, 0.64, 6.4, 53.0]):
    for irho0, rho0 in enumerate(rho0_vec):
        print(f"{igamma=},{irho0=}/{n}")
        # plot predictions for MLB model
        c_Xtest = np.array(
            [
                np.log(1.0 + xi),
                np.log(rho0),
                np.log(1.0 + 1000.0 * zeta),
                np.log(1.0 + gamma),
            ]
        )
        K_cross = np.array(
            [kernel(Xtrain_mat[i, :], c_Xtest, theta_opt) for i in range(n_train)]
        )
        K_cross = np.reshape(K_cross, (1, K_cross.shape[0]))
        # print(f"{K_cross=}")
        pred_log_load = (K_cross @ alpha)[0, 0]
        N11cr_mlb[irho0] = np.exp(pred_log_load)
        if args.load == "Nx":
            N11cr_tacs[irho0] = con.nondimCriticalGlobalAxialLoad(rho0, xi, gamma, zeta)
        else:
            N11cr_tacs[irho0] = con.nondimCriticalGlobalShearLoad(rho0, xi, gamma, zeta)

    plt.plot(
        rho0_vec,
        N11cr_mlb,
        "-",
        label=f"MLB-gamma={gamma:.2f}",
        color=colors[2 * igamma],
        linewidth=2,
    )
    plt.plot(
        rho0_vec,
        N11cr_tacs,
        "--",
        label=f"TACS-gamma={gamma:.2f}",
        color=colors[2 * igamma + 1],
    )

plt.xscale("log")
plt.yscale("log")
# plot it
plt.margins(x=0.05, y=0.05)
plt.xlabel(r"$\rho_0$")
if args.load == "Nx":
    plt.ylabel(r"$N_{11,cr}^*$")
else:
    plt.ylabel(r"$N_{12,cr}^*$")
plt.legend()
# plt.savefig("1-verify.png", dpi=400)
plt.show()
