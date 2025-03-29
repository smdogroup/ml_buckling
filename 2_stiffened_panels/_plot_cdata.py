import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data/Nx_raw_stiffened.csv")
arr = df.to_numpy()[:,np.array([1,2,3,4,7,8])].astype(np.double)
# print(f"{arr=}")
myarr = arr[:,0]
print(f"{myarr=}")
log_rho = np.log(arr[:,0])
log_gam = np.log(1.0 + arr[:,2])
log_rhostar = log_rho - 0.25 * log_gam
log_eig_FEA = np.log(arr[:,-2])
log_eig_CF = np.log(arr[:,-1])

plt.plot(log_rhostar, log_eig_FEA, 'o')
plt.plot(log_rhostar, log_eig_CF, 'o')
plt.show()