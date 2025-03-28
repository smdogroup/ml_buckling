import numpy as np
import tensorflow as tf

def smooth_relu(x, alpha):
    return np.log(1.0 + np.exp(alpha * x)) / alpha

def rational_quadratic_kernel(x, xp, th):
    # for fast kernel functions, x and xp shapes are
    # assume x is N x 1 x 3
    # assume xp is 1 x M x 3

    xd = x - xp
    # lengths = th[:3]
    # alph = th[3]
    length = th[0]
    alph = th[1]
    # xbar = xd / lengths
    xbar2 = tf.pow(xd / length, 2.0)
    # I should have made it length**2 here.. (correction)
    return np.power(tf.reduce_sum(xbar2, axis=-1) / 2.0 / alph + 1.0, -alph)

def buckling_RQ_kernel(x, xp, th):
    """custom kernel function no. 2"""
    RQ_term = rational_quadratic_kernel(x, xp, [th[3], th[4]])
    # print(f"{th[4]=}")
    rho_AL = smooth_relu(-x[:,:,0], th[0]) * smooth_relu(-xp[:,:,0], th[0])
    gam_LIN = x[:,:,2] * xp[:,:,2]

    return rho_AL + th[1] * gam_LIN + th[2] * RQ_term + th[5]