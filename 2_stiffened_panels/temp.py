from scipy.linalg import get_blas_funcs
print(get_blas_funcs("gemm"))

import numpy as np
np.__config__.show()
