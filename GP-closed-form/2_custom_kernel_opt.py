# demo a squared exponential GP on the axial dataset, measure interp and extrap error
import sys
import numpy as np
import pandas as pd
import os

sys.path.append("src/")
from eval_GPs import eval_GPs
from kernel_library import *

# baseline
# sigma_n = 1e-3
# axial = True; affine = True; log = True
# kernel = SE_kernel
# theta = np.array([8.0, 3.0, 1.0])

csv_file = "csvs/custom_kernel.csv"
kernel = custom_kernel1

# V1 - starting RMSE 0.55
# c_theta = np.array([1.0, 8.0, 4.0, 1.0, 1.0, 0.1])
# step_sizes = [
#     1.0, 1.0, 0.3, 0.2, 0.2, 0.1
# ]
# max_theta = np.array([10.0]*3 + [3.0]*3)
# min_theta = np.array([1.0, 1.0, 0.3, 0.2, 0.2, 0.1])

# V2 - starting RMSE 0.3112
c_theta = np.array([ 1. , 10. ,  4.3,  0.2,  1.2,  0.1])
step_sizes = [
    0.2, 0.2, 0.2, 0.2, 0.2, 0.1
]
max_theta = np.array([20.0]*3 + [3.0]*3)
min_theta = np.array([0.2]*5 + [0.1])

# not great result.. may need to improve kernels still, limiting is:
# ai 0.030, si 0.041, ssi 0.044 ae 0.200, se 0.190, sse 0.297 (limited mostly by sse)
# done, min_RMSE=0.2974936354174222 c_theta=array([ 1. , 10.2,  4.3,  0.2,  1. ,  0.1])

ct = 0
affine = True; log = True; sigma_n = 1e-2
  
def get_RMSE(theta, msg=""):
    # axial case
    axial_interp_RMSE, axial_extrap_RMSE = eval_GPs(
        kernel=kernel,
        theta=theta,
        n_trials=30,
        rand_seed=None, sigma_n=sigma_n,
        train_test_frac=0.8,
        axial=True, affine=affine, log=log,
        percentile=90.0,
        n_rho0=20, n_gamma=10, n_xi=5,
        can_print=False,
    )

    # shear case
    shear_interp_RMSE, shear_extrap_RMSE = eval_GPs(
        kernel=kernel,
        theta=theta,
        n_trials=30,
        rand_seed=None, sigma_n=sigma_n,
        train_test_frac=0.8,
        axial=False, affine=affine, log=log,
        percentile=90.0,
        n_rho0=20, n_gamma=10, n_xi=5,
        can_print=False,
    )

    # shear smoothed case
    shear_smooth_interp_RMSE, shear_smooth_extrap_RMSE = eval_GPs(
        kernel=kernel,
        theta=theta,
        n_trials=30,
        shear_ks_param=1.0,
        rand_seed=None, sigma_n=sigma_n,
        train_test_frac=0.8,
        axial=False, affine=affine, log=log,
        percentile=90.0,
        n_rho0=20, n_gamma=10, n_xi=5,
        can_print=False,
    )

    max_interp_RMSE = max([axial_interp_RMSE, shear_interp_RMSE, shear_smooth_interp_RMSE])
    max_extrap_RMSE = max([axial_extrap_RMSE, shear_extrap_RMSE, shear_smooth_extrap_RMSE])
    max_RMSE = max([max_interp_RMSE, max_extrap_RMSE])

    print(f"\t{msg}: ovr {max_RMSE:.4f};   ai {axial_interp_RMSE:.3f}, si {shear_interp_RMSE:.3f}, ssi {shear_smooth_interp_RMSE:.3f} " + \
          f"ae {axial_extrap_RMSE:.3f}, se {shear_extrap_RMSE:.3f}, sse {shear_smooth_extrap_RMSE:.3f}")
    
    return max_RMSE

min_RMSE = get_RMSE(c_theta)

# now do iterations for hyperparameter optimization
no_change_ct = 0
for iter in range(400):
    ind = iter % 6 
    step = step_sizes[ind]
    pert_theta = np.zeros((6,))
    pert_theta[ind] = step
    print(f"{iter} : {min_RMSE=}")

    # forward
    fw_theta = np.minimum(np.maximum(c_theta + pert_theta, min_theta), max_theta)
    fw_change = np.linalg.norm(fw_theta - c_theta) > 0.0
    if fw_change:
        fw_RMSE = get_RMSE(fw_theta, msg="fw")
        # print(f"\t{fw_RMSE=}")
    else:
        fw_RMSE = min_RMSE
        print(f"\tno fw change")

    # backwards
    bk_theta = np.minimum(np.maximum(c_theta - pert_theta, min_theta), max_theta)
    bk_change = np.linalg.norm(bk_theta - c_theta) > 0.0
    if bk_change:
        bk_RMSE = get_RMSE(bk_theta, msg="bk")
        # print(f"\t{bk_RMSE=}")
    else:
        bk_RMSE = min_RMSE
        print(f"\tno bk change")

    min_RMSE = np.min([bk_RMSE, fw_RMSE, min_RMSE])

    if fw_RMSE == min_RMSE and fw_change:
        c_theta += pert_theta
        # c_theta_list = list(c_theta)
        print(f"\tfw step {ind=}, RMSE={min_RMSE:.4f}, {c_theta[ind]-step}->{c_theta[ind]}, {c_theta=}")
        no_change_ct = 0
    elif bk_RMSE == min_RMSE and bk_change:
        c_theta -= pert_theta
        # c_theta_list = list(c_theta)
        print(f"\tbk step {ind=}, RMSE={min_RMSE:.4f}, {c_theta[ind]+step}->{c_theta[ind]}, {c_theta=}")
        no_change_ct = 0
    else:
        no_change_ct += 1
        if no_change_ct > 6:
            break

print(f"done, {min_RMSE=} {c_theta=}")