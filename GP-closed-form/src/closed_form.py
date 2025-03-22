import numpy as np
from scipy.optimize import fsolve

def ND_axial_buckling(rho0, gamma, xi):
    return min([
        (1+gamma) * m**2 / rho0**2 + rho0**2 / m**2 + 2 * xi for m in range(1, 50+1)
    ])

def ND_shear_Buckling(rho0, gamma, xi, shear_ks_param=None):
    def high_AR_resid(s2):
        s1 = (1.0 + 2.0 * s2 ** 2 * xi + s2 ** 4 + gamma) ** 0.25
        term1 = s2 ** 2 + s1 ** 2 + xi / 3
        term2 = (
            (3 + xi) / 9.0 + 4.0 / 3.0 * s1 ** 2 * xi + 4.0 / 3.0 * s1 ** 4
        ) ** 0.5
        return term1 - term2
    
    s2_bar = fsolve(high_AR_resid, 1.0)[0]
    s1_bar = (1.0 + 2.0 * s2_bar ** 2 * xi + s2_bar ** 4 + gamma) ** 0.25

    N12cr_highAR = (
        (
            1.0
            + gamma
            + s1_bar ** 4
            + 6 * s1_bar ** 2 * s2_bar ** 2
            + s2_bar ** 4
            + 2 * xi * (s1_bar ** 2 + s2_bar ** 2)
        )
        / 2.0
        / s1_bar ** 2
        / s2_bar
    )

    if shear_ks_param is not None: # does shear smoothing
        vals = np.array([1.0, rho0**(-2.0)])
        ks_max = 1.0 / shear_ks_param * np.log(np.sum(np.exp(shear_ks_param * vals)))
        return ks_max * N12cr_highAR
    else:
        return N12cr_highAR * max([1.0, rho0**(-2.0)])