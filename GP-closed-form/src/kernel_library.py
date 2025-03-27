import tensorflow as tf
import numpy as np
from kernel_utils import *

def SE_kernel(x, xp, th):
    # for fast kernel functions, x and xp shapes are
    # assume x is N x 1 x 3
    # assume xp is 1 x M x 3

    xd = x - xp
    lengths = th[:3]
    xbar = xd / lengths
    xbar2 = tf.pow(xbar, 2.0)
    # print(f"{xd.shape=}")
    return np.exp(-0.5 * tf.reduce_sum(xbar2, axis=-1))

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
    xbar2 = tf.pow(xd, 2.0) / length
    return np.power(tf.reduce_sum(xbar2, axis=-1) / 2.0 / alph + 1.0, -alph)
    
def matern_3_2_kernel(x, xp, th):
    # for fast kernel functions, x and xp shapes are
    # assume x is N x 1 x 3
    # assume xp is 1 x M x 3
    xd = x - xp
    xbar2 = tf.pow(xd, 2.0)
    norm = tf.sqrt(tf.reduce_mean(xbar2, axis=-1))
    rt3 = np.sqrt(3)
    return th[0]**2 * (1.0 + rt3 * norm / th[1]) * np.exp(-rt3 / th[1] * norm)
    
def matern_5_2_kernel(x, xp, th):
    # for fast kernel functions, x and xp shapes are
    # assume x is N x 1 x 3
    # assume xp is 1 x M x 3
    xd = x - xp
    xbar2 = tf.pow(xd, 2.0)
    norm = tf.sqrt(tf.reduce_mean(xbar2, axis=-1))
    rt5 = np.sqrt(5)
    return th[0]**2 * (1.0 + rt5 * norm / th[1] + 5.0 / 3.0 * norm * norm / th[1]**2) * \
            np.exp(-rt5 / th[1] * norm)

def buckling_SE_kernel(x, xp, th):
    """custom kernel function no. 1"""
    # x is N x 1 x 3, xp is 1 x M x 3
    # x_affine = x[:,:,0] - th[3] * x[:,:,1]
    # xp_affine = xp[:,:,0] - th[3] * xp[:,:,1]

    SE_term = SE_kernel(x, xp, th[:3])
    # print(f"{th[4]=}")
    rho_AL = smooth_relu(-x[:,:,0], th[3]) * smooth_relu(-xp[:,:,0], th[3])
    gam_LIN = x[:,:,1] * xp[:,:,1]
    # window_fact = smooth_relu(1.0 - np.abs(x[:,:,0]), th[3]) * smooth_relu(1.0 - np.abs(xp[:,:,0]), th[3])
    # SE_term2 = SE_kernel(x, xp, [0.3, 8.0, 4.0]) # try adding shorter length scale too, can help improve interp RMSE

    # V1
    # this kernel works well for axial + shear closed-form, but not a smoothed shear with ks = 1.0
    # need to improve to fit a smoothed shear profile
    return rho_AL + th[4] * gam_LIN + th[5] * SE_term + th[6]

def buckling_RQ_kernel(x, xp, th):
    """custom kernel function no. 2"""
    RQ_term = rational_quadratic_kernel(x, xp, [th[3], th[4]])
    # print(f"{th[4]=}")
    rho_AL = smooth_relu(-x[:,:,0], th[0]) * smooth_relu(-xp[:,:,0], th[0])
    gam_LIN = x[:,:,1] * xp[:,:,1]

    return rho_AL + th[1] * gam_LIN + th[2] * RQ_term + th[5]