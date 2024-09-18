import numpy as np
import matplotlib.pyplot as plt

# Define the function to plot
def f(x, y):
    s1 = 1.0
    s2 = 1.5
    case = 2
    if case == 1:
        return np.sin(np.pi * (x - s2 * (y - 0.5)) / s1) * np.sin(np.pi * y)
    elif case == 2:
        return (
            np.sin(np.pi * (x - s2 * (y - 0.5)) / s1)
            * np.sin(np.pi * y)
            * np.sin(np.pi * x)
        )


# Create grid and multivariate normal
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Create contour plot
plt.figure(figsize=(8, 6))
contour = plt.contour(X, Y, Z, levels=20, cmap="viridis")
plt.colorbar(contour)
plt.title("2D Contour Plot")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.show()
