# plot the primitive interpolation-based kernels (that do data fitting)


import numpy as np, matplotlib.pyplot as plt
import tensorflow as tf

import sys
sys.path.append("../src/")
# from kernel_library import SE_kernel


def SE_kernel(x, xp, th):
    # for fast kernel functions, x and xp shapes are
    # assume x is N x 1 x 3
    # assume xp is 1 x M x 3

    xd = x - xp
    lengths = th
    xbar = xd / lengths
    xbar2 = tf.pow(xbar, 2.0)
    # print(f"{xd.shape=}")
    return np.exp(-0.5 * tf.reduce_sum(xbar2))


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
    # I should have made it length**2 here.. (correction)
    return np.power(xbar2 / 2.0 / alph + 1.0, -alph)

   
def matern_3_2_kernel(x, xp, th):
    # for fast kernel functions, x and xp shapes are
    # assume x is N x 1 x 3
    # assume xp is 1 x M x 3
    xd = x - xp
    xbar2 = tf.pow(xd, 2.0)
    norm = tf.sqrt(xbar2)
    rt3 = np.sqrt(3)
    return th[0]**2 * (1.0 + rt3 * norm / th[1]) * np.exp(-rt3 / th[1] * norm)


def matern_5_2_kernel(x, xp, th):
    # for fast kernel functions, x and xp shapes are
    # assume x is N x 1 x 3
    # assume xp is 1 x M x 3
    xd = x - xp
    xbar2 = tf.pow(xd, 2.0)
    norm = tf.sqrt(xbar2)
    rt5 = np.sqrt(5)
    return th[0]**2 * (1.0 + rt5 * norm / th[1] + 5.0 / 3.0 * norm * norm / th[1]**2) * \
            np.exp(-rt5 / th[1] * norm)


def plot_1d_kernel(kernel_func, nsamples=100):
    npts = 101
    blues = plt.cm.jet(np.linspace(0, 1, nsamples))
    xvec = np.linspace(-3.0, 3.0, npts)
    mean = np.zeros((npts,))
    cov = np.array([[kernel_func(xvec[i], xvec[j]) for i in range(npts)] for j in range(npts)])
    samples = np.random.multivariate_normal(mean, cov, size=nsamples)

    for isample, sample in enumerate(samples):
        plt.plot(xvec, sample, color=blues[isample], alpha=0.5, linewidth=2)
    plt.show()

# th = 1.0
for th in [0.1, 1.0]:
    SE_kernel_func = lambda x1,x2 : SE_kernel(x1,x2,th)
    plot_1d_kernel(SE_kernel_func, nsamples=20 if th == 0.1 else 40)


# now rational quadratic
th = np.array([1.0, 0.01])
RQ_kernel_func = lambda x1,x2 : rational_quadratic_kernel(x1,x2,th)
plot_1d_kernel(RQ_kernel_func, nsamples=40)

# th = 1.0
# M32_kernel_func = lambda x1,x2 : matern_3_2_kernel(x1,x2,th)
# plot_1d_kernel(SE_kernel_func, nsamples=20 if th == 0.1 else 40)


# M32_kernel_func = lambda x1,x2 : matern_3_2_kernel(x1,x2,th)
# plot_1d_kernel(SE_kernel_func, nsamples=20 if th == 0.1 else 40)