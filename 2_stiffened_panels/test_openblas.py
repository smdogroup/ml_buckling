import numpy as np
import scipy.linalg as la
import os

# print(np.__config__.show())

A = np.random.rand(1000, 1000)
b = np.random.rand(1000)

# Check performance with OpenBLAS threading
os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "8"
x = la.solve(A, b)
