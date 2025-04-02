import numpy as np

def affine_transform(
    X:np.ndarray,
    is_log:bool=True,
) -> np.ndarray:
    # assumes X - [rho0, xi, gamma, zeta]
    if is_log:
        # print(f"{X[:,0]=}")
        # print(f"{X[:,2]=}")
        # exit()
        X[:,0] -= 0.25 * X[:,2]
    else:
        X[:,0] -= 0.25 * np.log(1.0+X[:,-1])
    return X