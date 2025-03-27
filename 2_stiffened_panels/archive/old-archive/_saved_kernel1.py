import numpy as np

# optimal hyperparameters from the hyperparameter optimization
# actual optimum for lower nMAP
# -------------------------------
# theta_opt = np.array([
#     0.3, 0.5878862050193697, 0.4043476413181697, 0.062024271977149964, 0.28832170210483743,
#     0.1194932804411767, 0.1123780262980141, 1.73775353254897, 0.8195497528948904,
#     1.0610251099528003, 1.1194429640030266, 0.2249190070996631, 0.09951153203496944,
#     0.08479401236107026, 0.09638992095113237, 0.9596691316041401
# ])

# modified for better extrapolation to high gamma (removing some SE parts)
# --------------------------------
theta_opt = np.array(
    [
        0.3,
        0.05,  # lower main SE term
        0.4043476413181697,
        1000.0,  # make gamma length scale huge to effectively remove that term for now.. 0.062024271977149964
        0.28832170210483743,
        0.1194932804411767,
        0.1123780262980141,
        1.0,  # lower the rho0 active zone back to to [-1,1]
        0.8195497528948904,
        1.0610251099528003,
        1.1194429640030266,
        0.2249190070996631,
        0.09951153203496944,
        0.08479401236107026,
        0.09638992095113237,
        0.9596691316041401,
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

    d0 = vec[0]  # xi direction
    d1 = vec[1]  # rho0 direction
    d2 = vec[2]  # zeta direction
    d3 = vec[3]  # gamma direction

    BL_kernel = theta[0] + soft_relu(-xp[1]) * soft_relu(-xq[1])
    # 0.02 was factor here before
    # SE_factor = theta[1] * np.exp(-0.5 * (d1**2 / theta[2]**2 + d3**2 / theta[3]**2))
    # delete gamma SE part
    SE_factor = theta[1] * np.exp(
        -0.5 * (d1 ** 2 / theta[2] ** 2 + d3 ** 2 / theta[3] ** 2)
    )
    SE_kernel2 = theta[4] * np.exp(
        -0.5 * (d0 ** 2 / theta[5] ** 2 + d2 ** 2 / theta[6] ** 2)
    )
    SE_kernel = (
        SE_factor
        * soft_relu(theta[7] - soft_abs(xp[1]))
        * soft_relu(theta[7] - soft_abs(xq[1]))  # * np.exp(-0.5 * d3 ** 2 / 9.0)
        * soft_relu(theta[8] - xp[3])
        * soft_relu(theta[8] - xq[3])  # correlate only low values of gamma together
    )
    gamma_kernel = theta[9] + theta[10] * xp[3] * xq[3]
    xi_kernel = theta[11] * xp[0] * xq[0] + theta[12] * (xp[0] * xq[0]) ** 2
    zeta_kernel = theta[13] * xp[2] * xq[2] + theta[14] * (xp[2] * xq[2]) ** 2
    sigma_n = theta[15]

    # multiplicative style results in very bad model where there are missing data points
    # inner_kernel = (BL_kernel + SE_kernel) * gamma_kernel**2 * (1.0 + xi_kernel) * (1.0 + zeta_kernel)

    # v1
    # inner_kernel = BL_kernel * gamma_kernel**2 + SE_kernel + xi_kernel + zeta_kernel + SE_kernel2

    # v2
    inner_kernel = (
        BL_kernel * gamma_kernel + SE_kernel + xi_kernel + zeta_kernel + SE_kernel2
    )
    return inner_kernel
