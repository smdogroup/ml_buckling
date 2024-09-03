import numpy as np
# optimal hyperparameters from the hyperparameter optimization
# actual optimum for lower nMAP
# -------------------------------
axial_theta_opt = np.array([0.29463415292059475, 1.0267438809351834, 0.08746650339467944, 
0.48466096307653356, 1.0545896944649038, 1.1208695945056195, 
0.6606359679565357, 0.10385825034258041, 1.7200477973124717, 
0.2720540795509361, 0.09746296712376498, 0.0972631230759857, 
0.09692913335000311, 0.9778323682800333])

# shear_theta_opt = ?
shear_theta_opt = axial_theta_opt

def relu(x):
    return max([0.0, x])

def soft_relu(x, rho=10):
    return 1.0 / rho * np.log(1 + np.exp(rho * x))

def soft_abs(x, rho=10):
    return 1.0 / rho * np.log(np.exp(rho*x) + np.exp(-rho*x))

def kernel(xp, xq, theta):
    # xp, xq are Nx1,Mx1 vectors (ln(1+xi), ln(rho_0), ln(1 + 10^3 * zeta), ln(1 + gamma))
    vec = xp - xq

    # d0 = vec[0] # xi direction
    d1 = vec[1] # rho0 direction
    # d2 = vec[2] # zeta direction
    # d3 = vec[3] # gamma direction

    gamma_rho_dist = theta[2] + theta[3] * xp[3] - xp[1]
    gamma_rho_dist_prime = theta[2] + theta[3] * xq[3] - xq[1]

    BL_kernel = theta[0] + theta[1] * soft_relu(gamma_rho_dist, 10) * soft_relu(gamma_rho_dist_prime, 10)
    gamma_kernel = theta[4] + theta[5] * xp[3] * xq[3]
    SE_kernel = theta[6] * np.exp(-0.5 * d1**2 / theta[7]**2)
    window_kernel = soft_relu(theta[8] - soft_abs(gamma_rho_dist, 10), 10) * \
                    soft_relu(theta[8] - soft_abs(gamma_rho_dist_prime, 10), 10)
    xi_linear = xp[0] * xq[0]
    xi_kernel = theta[9] * xi_linear + theta[10] * xi_linear**2
    zeta_linear = xp[2] * xq[2]
    zeta_kernel = theta[11] * zeta_linear + theta[12] * zeta_linear**2
    # sigma_n = theta[13]
    # 13 total hyperparameters of the model
    
    overall_kernel = BL_kernel * gamma_kernel + SE_kernel * window_kernel + xi_kernel + zeta_kernel
    return overall_kernel
