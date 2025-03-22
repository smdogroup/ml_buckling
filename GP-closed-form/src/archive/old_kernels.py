def custom_kernel1(x, xp, th):
    """custom kernel function no. 1"""
    # x is N x 1 x 3, xp is 1 x M x 3
    xd = x - xp
    lengths = th[:3]
    xbar = xd / lengths
    xbar2 = tf.pow(xbar, 2.0)

    rho_AL = smooth_relu(-x[:,:,0], th[3]) * smooth_relu(-xp[:,:,0], th[3])
    gam_LIN = x[:,:,1] * xp[:,:,1]
    SE_term = np.exp(-0.5 * tf.reduce_sum(xbar2, axis=-1))
    return rho_AL + th[4] * gam_LIN + th[5] * SE_term

def custom_kernel2(x, xp, th):
    """custom kernel function no. 2"""
    # x is N x 1 x 3, xp is 1 x M x 3
    xd = x - xp
    lengths = th[:3]
    xbar = xd / lengths
    xbar2 = tf.pow(xbar, 2.0)

    rho_AL = smooth_relu(-x[:,:,0], th[3]) * smooth_relu(-xp[:,:,0], th[3])
    gam_LIN = x[:,:,1] * xp[:,:,1]
    SE_term = np.exp(-0.5 * tf.reduce_sum(xbar2, axis=-1))
    # in this case, try to only apply SE for some x_rho0 > -1 for instance
    window_fact = smooth_relu(x[:,:,0] + th[4], th[5]) * smooth_relu(xp[:,:,0] + th[4], th[5])
    return rho_AL + th[6] * gam_LIN + th[7] * SE_term * window_fact

def custom_kernel3(x, xp, th):
    """custom kernel function no. 3"""
    # x is N x 1 x 3, xp is 1 x M x 3
    xd = x - xp
    lengths = th[:3]
    xbar = xd / lengths
    xbar2 = tf.pow(xbar, 2.0)

    rho_AL = smooth_relu(-x[:,:,0], th[3]) * smooth_relu(-xp[:,:,0], th[3])
    gam_LIN = x[:,:,1] * xp[:,:,1]
    xi = x[:,:,2] * xp[:,:,2] + 0.1 * (x[:,:,2] * xp[:,:,2])**2
    # SE_term = np.exp(-0.5 * tf.reduce_sum(xbar2, axis=-1))
    # in this case, try to only apply SE for some x_rho0 > -1 for instance
    # window_fact = smooth_relu(x[:,:,0] + th[4], th[5]) * smooth_relu(xp[:,:,0] + th[4], th[5])
    return rho_AL + th[4] * gam_LIN + th[5] * xi

elif args.kernel == 5:
    # V2: 
    # th = np.array([8.0, 8.0, 4.0, 0.5, 1.0, 0.1])
    # V3: trying to remove gamma linear term 
    # th = np.array([8.0, 8.0, 4.0, 0.5, 0.0, 0.1])
elif args.kernel == 6:
    kernel = custom_kernel2
    # V1: 0.087 axial, 0.5 shear (smoothed or not)
    th = np.array([1.0, 10.0, 4.3, 0.2, 1.0, 1.5, 1.2, 0.1])

elif args.kernel == 7:
    kernel = custom_kernel3
    # V1: 0.087 axial, 0.5 shear (smoothed or not)
    th = np.array([1.0, 10.0, 4.3, 0.1, 1.0, 1.0])