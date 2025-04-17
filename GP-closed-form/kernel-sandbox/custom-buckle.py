# 1D kernel plotting utility
import numpy as np, matplotlib.pyplot as plt
import niceplots

np.random.rand(123)

def plot_1d_kernel(kernel_func, nsamples=100):
    npts = 100
    blues = plt.cm.jet(np.linspace(0, 1, nsamples))
    xvec = np.linspace(-3.0, 3.0, npts)
    mean = np.zeros((npts,))
    cov = np.array([[kernel_func(xvec[i], xvec[j]) for i in range(npts)] for j in range(npts)])
    samples = np.random.multivariate_normal(mean, cov, size=nsamples)

    plt.style.use(niceplots.get_style())
    for isample, sample in enumerate(samples):
        plt.plot(xvec, sample, color=blues[isample], alpha=0.7, linewidth=2)
    # plt.show()
    plt.xlabel(r"$\boldsymbol{x}^*$", fontsize=24)
    plt.ylabel(r"$\boldsymbol{ \overline{N}_{ij,cr}(x) }$", fontsize=24)
    plt.savefig("custom-buckle.svg", dpi=400)

def smooth_relu(x, alpha):
    return np.log(1.0 + np.exp(alpha * x)) / alpha

def RQ(x, xp, den, exp):
    xd = x - xp
    return np.power( (1 + np.dot(xd, xd) / den), exp)

# axial kernel from eqn 9 of closed-form extrap
my_kernel = lambda x, xp : 0.8 + smooth_relu(-x, 10.0) * smooth_relu(-xp, 10.0) #+ 0.1 * RQ(x, xp, 2.0, -0.1)
plot_1d_kernel(my_kernel, nsamples=15) #10