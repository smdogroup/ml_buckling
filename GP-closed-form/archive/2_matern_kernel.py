# demo a squared exponential GP on the axial dataset, measure interp and extrap error
import sys
import numpy as np
import pandas as pd
import os

sys.path.append("src/")
from eval_GPs import eval_GPs
from kernel_library import *

# csv_file = "csvs/matern32_kernel.csv"
# kernel = matern_3_2_kernel

csv_file = "csvs/matern52_kernel.csv"
kernel = matern_5_2_kernel

ct = 0
for affine in [True, False]:
    for log in [True, False]:
        for sigma_n in [1e-2, 1e-4]:
            for sigma in [0.1, 1.0]:
                for L in [1.0, 4.0, 8.0]:
                    theta = np.array([sigma, L])
                    ct += 1

                    # axial case
                    axial_interp_RMSE, axial_extrap_RMSE = eval_GPs(
                        kernel=kernel,
                        theta=theta,
                        n_trials=30,
                        rand_seed=None, sigma_n=sigma_n,
                        train_test_frac=0.8,
                        axial=True, affine=affine, log=log,
                        percentile=90.0,
                        n_rho0=20, n_gamma=10, n_xi=5
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
                        n_rho0=20, n_gamma=10, n_xi=5
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
                        n_rho0=20, n_gamma=10, n_xi=5
                    )

                    max_extrap_RMSE = max([axial_extrap_RMSE, shear_extrap_RMSE, shear_smooth_extrap_RMSE])
                    

                    data_dict = {
                    'affine' : [affine],
                    'log' : [log],
                    'sigma_n' : [sigma_n],
                    'sigma' : [sigma],
                    'L' : [L],
                    'N11_int_RMSE' : [axial_interp_RMSE],
                    'N12_int_RMSE' : [shear_interp_RMSE],
                    'N11_ext_RMSE' : [axial_extrap_RMSE],
                    'N12_ext_RMSE' : [shear_extrap_RMSE],
                    'N12_smooth_ext_RMSE' : [shear_smooth_extrap_RMSE],
                    'max_ext_RMSE' : [max_extrap_RMSE],
                    }
                    df = pd.DataFrame(data_dict)
                    df.to_csv(csv_file, index=False, float_format="%.4f", 
                                mode='w' if ct == 1 else 'a', header=ct == 1)


