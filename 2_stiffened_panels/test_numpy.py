import numpy as np
import scipy.linalg as la
import os

# print(np.__config__.show())

N = 5000
A = np.random.rand(N, N)
b = np.random.rand(N)

# Check performance with OpenBLAS threading
os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "8"

x = np.linalg.solve(A, b)
# x = la.solve(A, b)
