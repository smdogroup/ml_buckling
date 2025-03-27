import matplotlib.pyplot as plt
import niceplots

# plot the extrapolation error with increasing kfold number
# for buckling+RQ with axial+affine+log and shear+noaffine+log datasets

kfolds = [5, 10, 20, 40, 80]
axial_extrap_err = [0.8796, 0.9959, 0.9966, 0.9612, 0.8796]
shear_affine_extrap_err = [0.9865] + [0.9]*4 # still waiting on this noaffine vs data
shear_noaffine_extrap_err = [0.8202, 0.8943, 0.9541, 0.9358, 0.8453] # affine version

plt.style.use(niceplots.get_style())
plt.plot(kfolds, axial_extrap_err, 'o-', label="axial-affine")
# plt.plot(kfolds, shear_affine_extrap_err, 'o-', label="shear-affine")
plt.plot(kfolds, shear_noaffine_extrap_err, 'o-', label="shear-noaffine")
plt.xscale('log')
plt.margins(x=0.05, y=0.05)
# plt.xticks([5, 10, 40, 80])
plt.xticks(kfolds, kfolds)

plt.xlabel(r"num $k$-folds")
plt.ylabel(r"$R^2$ - Extrapolation")

plt.legend()
# plt.show()
plt.savefig("output/kfold-study.svg", dpi=400)