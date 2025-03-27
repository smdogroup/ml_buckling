# 1D kernel plotting utility
import numpy as np, matplotlib.pyplot as plt

def plot_1d_kernel(kernel_func, nsamples=100):
    npts = 100
    blues = plt.cm.jet(np.linspace(0, 1, nsamples))
    xvec = np.linspace(-3.0, 3.0, npts)
    mean = np.zeros((npts,))
    cov = np.array([[kernel_func(xvec[i], xvec[j]) for i in range(npts)] for j in range(npts)])
    samples = np.random.multivariate_normal(mean, cov, size=nsamples)

    for isample, sample in enumerate(samples):
        plt.plot(xvec, sample, color=blues[isample], alpha=0.7, linewidth=2)
    plt.show()

def smooth_relu(x, alpha):
    return np.log(1.0 + np.exp(alpha * x)) / alpha

alpha = 3.0
my_kernel = lambda x, xp: smooth_relu(-x, alpha) * smooth_relu(-xp, alpha)
plot_1d_kernel(my_kernel, nsamples=10)