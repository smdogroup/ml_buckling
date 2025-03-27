import numpy as np

def affine_transform(
    X:np.ndarray,
    is_log:bool=True,
) -> np.ndarray:
    # assumes X - [xi, rho0, zeta, gamma]
    if is_log:
        X[:,1] -= 0.25 * X[:,-1]
    else:
        X[:,1] -= 0.25 * np.log(1.0+X[:,-1])
    return X