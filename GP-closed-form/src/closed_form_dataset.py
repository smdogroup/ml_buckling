import numpy as np
from typing import TYPE_CHECKING, List
from closed_form import ND_axial_buckling, ND_shear_Buckling

"""
Generate dataset on (rho_0, gamma, xi) => axial and shear ND buckling loads
"""
def get_closed_form_data(
    axial:bool=True, # if False is shear
    log_transform:bool=False,
    affine_transform:bool=False,
    include_extrapolation:bool=False,
    include_interpolation:bool=True,
    nan_extrapolation:bool=False, # helpful for plotting purposes
    nan_interpolation:bool=False, # helpful for plotting purposes
    log_xi:bool=False,
    shear_ks_param:float=None, # if None no shear smoothing, else some shear smoothing
    rho0_bounds:list=[0.2, 10.0],
    gamma_bounds:list=[0.0, 15.0],
    xi_bounds:list=[0.3, 1.5],
    n_rho0:int=10,
    n_gamma:int=10,
    n_xi:int=5
):
    
    rho0_vec = np.geomspace(rho0_bounds[0], rho0_bounds[1], n_rho0)
    gamma_vec = np.geomspace(1 + gamma_bounds[0], 1 + gamma_bounds[1], n_gamma) - 1.0
    xi_vec = np.linspace(xi_bounds[0], xi_bounds[1], n_xi)

    # print(f"{rho0_vec.shape=} {gamma_vec.shape=} {xi_vec.shape=}")

    # store each of the inputs in a list, will convert to np.ndarray later
    x_list = []
    y_list = []

    for rho0 in rho0_vec:
        for gamma in gamma_vec:
            for xi in xi_vec:
        
                # extrapolation zone, rho0 = 0.2 at gamma = 0 and rho_0 approx 3 at 1+gamma = 16
                in_interp_zone = np.log(1.0 + gamma) - np.log(rho0 / 0.2) < 0.0
                keep_data = (in_interp_zone and include_interpolation) \
                      or (not(in_interp_zone) and include_extrapolation)

                if keep_data:
                    # append inputs
                    x_list += [[rho0, gamma, xi]]

                    # compute the buckling loads
                    if axial:
                        buckling_load = ND_axial_buckling(rho0, gamma, xi)
                    else: # shear
                        buckling_load = ND_shear_Buckling(rho0, gamma, xi, shear_ks_param=shear_ks_param)

                    # change value to nan if trying to exclude extrapolation region in plot
                    if not(in_interp_zone) and nan_extrapolation:
                        buckling_load = np.nan
                    if in_interp_zone and nan_interpolation:
                        buckling_load = np.nan

                    y_list += [buckling_load]

    X = np.array(x_list)
    Y = np.array(y_list)

    # now do affine transform
    if affine_transform:
        X[:,0] /= np.power(1.0 + X[:,1], 0.25)

    # now do log transform
    if log_transform:
        X[:,0] = np.log(X[:,0])
        X[:,1] = np.log(1.0 + X[:,1])
        if log_xi:
            X[:,2] = np.log(1 + X[:,2])
        Y = np.log(Y)

    return X, Y

def nan_extrap_data(X, Y, 
                    log=True, affine=True, 
                    nan_extrap=True):
    nan_interp = not(nan_extrap)

    if log:
        gamma = np.exp(X[:,1]) - 1.0
    else:
        gamma = X[:,1]
    
    if affine and log:
        rho0_star = np.exp(X[:,0])
        rho0 = rho0_star * np.power(1.0 + gamma, 0.25)
    elif not(affine) and log:
        rho0 = np.exp(X[:,0])
    elif affine and not(log):
        rho0_star = X[:,0]
        rho0 = rho0_star * np.power(1.0 + gamma, 0.25)
    else: # not affine and not log
        rho0 = X[:,0]

    Y2 = Y.copy()

    in_interp_zone = np.log(1.0 + gamma) - np.log(rho0 / 0.2) < 0.0
    if nan_interp:
        Y2[in_interp_zone] = np.nan
    if nan_extrap:
        Y2[np.logical_not(in_interp_zone)] = np.nan
    return Y2
    

def split_data(X, Y, train_test_split:float=0.9):
    """split the data for training and testing"""
    assert 0 < train_test_split < 1, "train_test_split must be between 0 and 1"
    assert len(X) == len(Y), "X and Y must have the same length"

    N = len(X)
    indices = np.arange(N)
    np.random.shuffle(indices)  # Shuffle indices

    split_idx = int(N * train_test_split)  # Compute split index

    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    X_train, Y_train = X[train_indices], Y[train_indices]
    X_test, Y_test = X[test_indices], Y[test_indices]

    return X_train, Y_train, X_test, Y_test