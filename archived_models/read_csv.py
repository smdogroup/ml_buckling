import numpy as np

arr = np.loadtxt("axialGP.csv", skiprows=1, delimiter=",")
print(f"{arr=}")