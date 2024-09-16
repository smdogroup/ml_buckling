import numpy as np

# optimal hyperparameters from the hyperparameter optimization
# actual optimum for lower nMAP
# -------------------------------
axial_theta_opt = np.array(
    [
        0.29043362272636686,
        0.2767385687947649,
        1.0783621945553847,
        0.637338057553291,
        1.140876330048335,
        0.0892471492240224,
        0.0908920116313985,
        0.08956373647356093,
        0.6939463641124705,
        0.18098983021747583,
        0.7079870474480225,
        1.7479508137685653,
        # 0.18098983021747583, 400.0, 1.7479508137685653,
        0.9341625075293126,
    ]
)

# TODO : re-optimize shear theta
shear_theta_opt = axial_theta_opt


# was trying to limit theta[8] earlier rho_0 window => didn't help that much made it worse
# although still need to add higher AR datapoints to both axial + shear otherwise SE kernel
# drops off suddenly.. and it learns the low rho_0 slopes in


def relu(x):
    return max([0.0, x])


def soft_relu(x, rho=10):
    return 1.0 / rho * np.log(1 + np.exp(rho * x))


def soft_abs(x, rho=10):
    return 1.0 / rho * np.log(np.exp(rho * x) + np.exp(-rho * x))


def kernel(xp, xq, theta):
    # xp, xq are Nx1,Mx1 vectors (ln(1+xi), ln(rho_0), ln(1 + 10^3 * zeta), ln(1 + gamma))
    vec = xp - xq

    # d0 = vec[0] # xi direction
    d1 = vec[1]  # rho0 direction
    # d2 = vec[2] # zeta direction
    d3 = vec[3]  # gamma direction

    gamma_rho_dist = xp[1] - theta[0] * xp[3]
    gamma_rho_dist_prime = xq[1] - theta[0] * xq[3]

    dgr = gamma_rho_dist - gamma_rho_dist_prime

    BL_kernel = theta[1] + theta[2] * soft_relu(-gamma_rho_dist, 10) * soft_relu(
        -gamma_rho_dist_prime, 10
    )
    gamma_kernel = 1.0 + theta[3] * xp[3] * xq[3]
    xi_linear = xp[0] * xq[0]
    xi_kernel = 1.0 + theta[4] * xi_linear + theta[5] * xi_linear ** 2
    zeta_linear = xp[2] * xq[2]
    zeta_kernel = 1.0 + theta[6] * zeta_linear + theta[7] * zeta_linear ** 2
    SE_kernel = theta[8] * np.exp(
        -0.5 * dgr ** 2 / theta[9] ** 2 - 0.5 * d3 ** 2 / theta[10] ** 2
    )
    window_kernel = soft_relu(theta[11] - soft_abs(gamma_rho_dist, 10), 10) * soft_relu(
        theta[11] - soft_abs(gamma_rho_dist_prime, 10), 10
    )
    # sigma_n = theta[12]
    # 13 total hyperparameters of the model

    overall_kernel = (
        BL_kernel * gamma_kernel * xi_kernel * zeta_kernel + SE_kernel * window_kernel
    )
    return overall_kernel
