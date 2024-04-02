import matplotlib.pyplot as plt
import pandas as pd
import niceplots

df = pd.read_csv("1_hstudy.csv")
h_w = df['h_w'].to_numpy()
tacs_eigvals = df['tacs_eig'].to_numpy()
CF_eigvals = df['CF_eig'].to_numpy()

plt.style.use(niceplots.get_style())
plt.plot(h_w, tacs_eigvals, "o", label="FEA-FSDT")
plt.plot(h_w, CF_eigvals, label="CF-CPT")
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$h_w$")
plt.ylabel(r"$\lambda_{min}^*$")
plt.legend()
plt.savefig("1_hstudy.png", dpi=400)