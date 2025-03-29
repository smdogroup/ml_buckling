import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data/Nx_raw_stiffened.csv")
arr = df.to_numpy()[:,1:]
print(f"{arr=}")
log_rho = np.log(arr[:,0])
log_gam = np.log(1.0 + arr[:,2])
log_rhostar = log_rho - 0.25 * log_gam
log_eig_FEA = np.log(arr[:,-3])
log_eig_CF = np.log(arr[:,-2])

plt.plot(log_rhostar, log_eig_FEA, 'ko')
plt.plot(log_rhostar, log_eig_CF, 'bo')
plt.show()