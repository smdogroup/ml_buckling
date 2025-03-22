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
    lengths = th[:3]
    alph = th[3]
    xbar = xd / lengths
    xbar2 = tf.pow(xbar, 2.0)
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

def custom_kernel1(x, xp, th):
    """custom kernel function no. 1"""
    # x is N x 1 x 3, xp is 1 x M x 3
    x_affine = x[:,:,0] - th[3] * x[:,:,1]
    xp_affine = xp[:,:,0] - th[3] * xp[:,:,1]

    xd = x - xp
    lengths = th[:3]
    xbar = xd[:,:,1:] / lengths[1:]
    xbar2 = tf.pow(xbar, 2.0)
    xbar2_sum = tf.reduce_sum(xbar2, axis=-1)
    xbar2_sum2 = xbar2_sum + tf.pow((x_affine - xp_affine) / lengths[0], 2.0)

    rho_AL = smooth_relu(-x_affine, th[4]) * smooth_relu(-xp_affine, th[4])
    gam_LIN = x[:,:,1] * xp[:,:,1]
    SE_term = np.exp(-0.5 * xbar2_sum2)
    return rho_AL + th[5] * gam_LIN + th[6] * SE_term