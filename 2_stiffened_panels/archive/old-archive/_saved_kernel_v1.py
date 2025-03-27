import numpy as np

# optimal hyperparameters from the hyperparameter optimization
# theta_opt = np.array([
#     0.29826658577705645, 0.7185836683907613, 0.5227143922532845, 0.6284872734685776,
#     1.9135627407198488, 0.9932663026728726, 1.215272984407997, 1.131107170054227,
#     0.26204218577436783, 0.09904271405413381, 0.9859641343064356
# ])
theta_opt = np.array(
    [
        0.2983788970125638,
        0.7368171728101005,
        0.4920863192095234,
        0.5910059723763343,
        1.9191631767004782,
        0.9937036838855656,
        1.2998931465359926,
        1.1507545094431129,
        0.32174014407471807,
        0.0991047384484708,
        0.9854386085274075,
    ]
)


def relu(x):
    return max([0.0, x])


def soft_relu(x, rho=10):
    return 1.0 / rho * np.log(1 + np.exp(rho * x))


def soft_abs(x, rho=10):
    return 1.0 / rho * np.log(np.exp(rho * x) + np.exp(-rho * x))


def kernel(xp, xq, theta, include_sigma=True):
    # xp, xq are Nx1,Mx1 vectors (ln(1+xi), ln(rho_0), ln(1 + 10^3 * zeta), ln(1 + gamma))
    vec = xp - xq

    d1 = vec[1]  # rho0 direction
    d2 = vec[2]  # zeta direction
    d3 = vec[3]  # gamma direction

    BL_kernel = theta[0] + soft_relu(-xp[1]) * soft_relu(-xq[1])
    # 0.02 was factor here before
    SE_factor = theta[1] * np.exp(
        -0.5 * (d1 ** 2 / theta[2] ** 2 + d3 ** 2 / theta[3] ** 2)
    )
    SE_kernel = (
        SE_factor
        * soft_relu(theta[4] - soft_abs(xp[1]))
        * soft_relu(theta[4] - soft_abs(xq[1]))  # * np.exp(-0.5 * d3 ** 2 / 9.0)
        * soft_relu(theta[5] - xp[3])
        * soft_relu(theta[5] - xq[3])  # correlate only low values of gamma together
    )
    gamma_kernel = theta[6] + theta[7] * xp[3] * xq[3]
    xi_kernel = theta[8] * xp[0] * xq[0]
    zeta_kernel = theta[9] * xp[2] * xq[2]
    sigma_n = theta[10]

    inner_kernel = BL_kernel * gamma_kernel ** 2 + SE_kernel + xi_kernel + zeta_kernel
    # return inner_kernel + sigma_n**2 * include_sigma
    return inner_kernel
