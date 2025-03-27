import numpy as np, matplotlib.pyplot as plt

relu = lambda x : max([0.0, x])
unit_step = lambda x : 1.0 * (x > 0.0)

def kernel(_x, _xp):
    x = _x[0]; y = _x[1]
    xp = _xp[0]; yp = _xp[1]
    
    

x_vec = np.linspace(-1.0, 1.0, 10)
y_vec = np.linspace(-1.0, 1.0, 10)

X, Y = np.meshgrid(x_vec, y_vec)
X_plot = np.zeros((100, 2))
for i in range(10):
    for j in range(10):
        n = 10 * i + j
        X_plot[n, 0] = x_vec[i]
        X_plot[n, 1] = y_vec[j]

cov = np.array(
    [[kernel(X_plot[i, :], X_plot[j, :]) for i in range(100)] for j in range(100)]
)
# random samples
mean = np.zeros((100,))
Y_arr = np.random.multivariate_normal(mean, cov, 4)
for i in range(4):
    cY_arr = Y_arr[i]
    X1 = X_plot[:, 0].reshape((10, 10))
    X2 = X_plot[:, 1].reshape((10, 10))
    Ym = cY_arr.reshape((10, 10))
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X1, X2, Ym)
    plt.show()