def custom_kernel1(x, xp, th):
    """custom kernel function no. 1"""
    # x is N x 1 x 3, xp is 1 x M x 3
    # x_affine = x[:,:,0] - th[3] * x[:,:,1]
    # xp_affine = xp[:,:,0] - th[3] * xp[:,:,1]

    SE_term = SE_kernel(x, xp, th[:3])
    # print(f"{th[4]=}")
    rho_AL = smooth_relu(-x[:,:,0], th[3]) * smooth_relu(-xp[:,:,0], th[3])
    gam_LIN = x[:,:,1] * xp[:,:,1]
    window_fact = smooth_relu(1.0 - np.abs(x[:,:,0]), th[3]) * smooth_relu(1.0 - np.abs(xp[:,:,0]), th[3])
    SE_term2 = SE_kernel(x, xp, [0.3, 8.0, 4.0]) # try adding shorter length scale too, can help improve interp RMSE

    # V1
    # this kernel works well for axial + shear closed-form, but not a smoothed shear with ks = 1.0
    # need to improve to fit a smoothed shear profile
    return rho_AL + th[4] * gam_LIN + th[5] * SE_term 

    # V2
    # return rho_AL + th[5] * gam_LIN + th[6] * SE_term * window_fact #+ SE_term2

    # V3
    # return rho_AL + th[5] * gam_LIN + th[6] * SE_term + matern_3_2_kernel(x, xp, [1.0, 8.0])

def custom_kernel2(x, xp, th):
    """custom kernel function no. 2"""
    # x is N x 1 x 3, xp is 1 x M x 3
    matern_term = matern_3_2_kernel(x, xp, [1.0, th[0]])
    rho_AL = smooth_relu(-x[:,:,0], th[1]) * smooth_relu(-xp[:,:,0], th[1])
    gam_LIN = x[:,:,1] * xp[:,:,1]
    return rho_AL + th[2] * gam_LIN + th[3] * matern_term

def custom_kernel3(x, xp, th):
    """custom kernel function no. 3, intended for shear buckling especially"""
    # x is N x 1 x 3, xp is 1 x M x 3
    SE_term = SE_kernel(x, xp, th[:3])
    # print(f"{th[4]=}")
    my_unit_step = lambda xin : smooth_unit_step(xin, th[4])
    basis = lambda x,p : (1 - p * x) * my_unit_step(-(x+1)) + \
        1 * my_unit_step(x-1) + (1 + p/2 * (1-x)) * (my_unit_step(x+1) - my_unit_step(x-1))
    rho_piecewise = basis(x[:,:,0],2) * basis(xp[:,:,0],2) + 1.0
    gam_LIN = x[:,:,1] * xp[:,:,1]
    return rho_piecewise + th[5] * gam_LIN + th[6] * SE_term 

def custom_kernel4(x, xp, th):
    """custom kernel function no. 4, intended for shear buckling especially"""
    # x is N x 1 x 3, xp is 1 x M x 3
    SE_term = SE_kernel(x, xp, th[:3])
    # print(f"{th[4]=}")
    basis = lambda x, p, ks : 1 + np.log(1.0 + np.exp(-p * ks * x)) / ks
    # could make ks a hyperparameter later!
    ks_kernel2 = lambda x, xp : basis(x, 2, th[4]) * basis(x,2,th[4]) + 0.2
    gam_LIN = x[:,:,1] * xp[:,:,1]
    return ks_kernel2 + th[5] * gam_LIN + th[6] * SE_term 

def custom_kernel5(x, xp, th):
    """custom kernel function no. 5 intended for not log scale"""
    # x is N x 1 x 3, xp is 1 x M x 3
    SE_term = SE_kernel(x, xp, th[:3])
    # could make ks a hyperparameter later!
    custom_term = 1.0 + np.sqrt(1.0 + x[:,:,1]) * np.maximum(1.0, x[:,:,0]**(-2)) * \
          np.sqrt(1.0 + xp[:,:,1]) * np.maximum(1.0, xp[:,:,0]**(-2))
    return 1.0 + custom_term + th[6] * SE_term 

def custom_kernel6(x, xp, th):
    """custom kernel function no. 1"""
    # x is N x 1 x 3, xp is 1 x M x 3
    # x_affine = x[:,:,0] - th[3] * x[:,:,1]
    # xp_affine = xp[:,:,0] - th[3] * xp[:,:,1]

    SE_term = SE_kernel(x, xp, th[:3])
    # print(f"{th[4]=}")
    rho_AL = smooth_relu(-x[:,:,0], th[3]) * smooth_relu(-xp[:,:,0], th[3])
    gam_LIN = x[:,:,1] * xp[:,:,1]
    window_fact = smooth_relu(1.0 - np.abs(x[:,:,0]), th[3]) * smooth_relu(1.0 - np.abs(xp[:,:,0]), th[3])
    SE_term2 = SE_kernel(x, xp, [0.3, 8.0, 4.0]) # try adding shorter length scale too, can help improve interp RMSE

    # V1
    # this kernel works well for axial + shear closed-form, but not a smoothed shear with ks = 1.0
    # need to improve to fit a smoothed shear profile
    return rho_AL + th[4] * gam_LIN + th[5] * SE_term 

# this kernel was bad
# elif args.kernel == 6:
#     kernel = custom_kernel2
#     # matern_length, smooth_relu, gamma_term, matern_term
#     th = np.array([8.0, 1.0, 1.0, 0.1])

# really should use a commercial hyperparameter optimize