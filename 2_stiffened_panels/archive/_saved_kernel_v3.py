import numpy as np

# optimal hyperparameters from the hyperparameter optimization
# actual optimum for lower nMAP
# -------------------------------
axial_theta_opt = np.array(
    [
        0.2958297749925822,
        1.0239945828006594,
        0.07168299610058654,
        0.48562514466224493,
        1.0506070941509835,
        1.1177330502913305,
        0.6848138967385287,
        0.11354534126709757,
        1.8907798782384087,
        0.27212866705427174,
        0.09785062102760544,
        0.09714329007582441,
        0.09680515694078637,
        0.9767787788259286,
    ]
)

# shear_theta_opt = ?
shear_theta_opt = np.array(
    [
        0.29200307826637956,
        1.0333801491678516,
        0.4419183782181758,
        0.4384412893181439,
        1.1126466447969539,
        1.102949479208879,
        0.9606247321578347,
        0.057500507690135844,
        1.958551716347745,
        0.32550807276345095,
        0.09630193606084558,
        0.0966842564187835,
        0.09626828403078751,
        0.9621053901376977,
    ]
)


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
    # d3 = vec[3] # gamma direction

    gamma_rho_dist = theta[2] + theta[3] * xp[3] - xp[1]
    gamma_rho_dist_prime = theta[2] + theta[3] * xq[3] - xq[1]

    BL_kernel = theta[0] + theta[1] * soft_relu(gamma_rho_dist, 10) * soft_relu(
        gamma_rho_dist_prime, 10
    )
    gamma_kernel = theta[4] + theta[5] * xp[3] * xq[3]
    SE_kernel = theta[6] * np.exp(-0.5 * d1 ** 2 / theta[7] ** 2)
    window_kernel = soft_relu(theta[8] - soft_abs(gamma_rho_dist, 10), 10) * soft_relu(
        theta[8] - soft_abs(gamma_rho_dist_prime, 10), 10
    )
    xi_linear = xp[0] * xq[0]
    xi_kernel = theta[9] * xi_linear + theta[10] * xi_linear ** 2
    zeta_linear = xp[2] * xq[2]
    zeta_kernel = theta[11] * zeta_linear + theta[12] * zeta_linear ** 2
    # sigma_n = theta[13]
    # 13 total hyperparameters of the model

    overall_kernel = (
        BL_kernel * gamma_kernel + SE_kernel * window_kernel + xi_kernel + zeta_kernel
    )
    return overall_kernel
