import numpy as np

relu = lambda x : max([0.0, x])
unit_step = lambda x : 1.0 * (x > 0.0)

def smooth_relu(x, alpha):
    return np.log(1.0 + np.exp(alpha * x)) / alpha

def smooth_abs(x, alpha):
    return np.log(np.exp(-alpha * x) + np.exp(alpha * x)) / alpha

def smooth_unit_step(x, alpha):
    # not symmetric and breaks kernel functions
    return 1.0 / (1.0 + np.exp(-alpha * x))

def smooth_unit_step2(x, alpha):
    # hopefully better numerical stability for kernel functions (since sym about xx' < 0 and xx' > 0)
    return 0.5 * (1.0 + np.tanh(alpha * x))