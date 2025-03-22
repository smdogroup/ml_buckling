# Best GP model for closed-form solutions

* goal here is to make a section in the paper that identifies the best kernel functions and GP model for fitting the closed-form solution
* evaluate both test interpolation RMSE and test extrapolation RMSE where some low rho_0, high gamma data are left out
    * useful exercise here is that I actually have the missing data for the closed-form in this regime, so can check extrapolation error
    * then once find the best kernel functions or GP model use this for the stiffened panel kernels
    * consider low rho_0 and high rho_0 extrapolations?
* show the following equations:
    $$ \rho_0 \rightarrow 0 : \quad \overline{N}_{11}^{cr} = \frac{1+\gamma}{\rho_0^2} = \frac{\sqrt{1+\gamma}}{(\rho_0^*)^2} $$
    $$ \rho_0 \rightarrow \infty: \quad \overline{N}_{11}^{cr} = 2 \sqrt{1 + \gamma} + 2 \xi $$
    and in general at intermediate aspect ratios:
    $$ \overline{N}_{11}^{cr} = \min_{m \in N} \left[ \frac{1+\gamma}{\rho_0^2} \cdot m^2 + \frac{\rho_0^2}{m^2} + 2 \xi \right] = \sqrt{1+\gamma} \cdot \min_{m \in N} \left[ \frac{m^2}{(\rho_0^*)^2} + \frac{(\rho_0^*)^2}{m^2} + 2 \xi \right] $$
    * thus the $\rho_0 \rightarrow \rho_0^*$ affine transform greatly simplifies the equations so the same exponent or log-transform slope is used for $1+\gamma$ at low and high aspect ratios. Then we only need a regular additive linear kernel, no kernel multiplication of $k_{\gamma}$

* purpose of this exercise:
    * fit GP surrogate models to the closed-form solution $\overline{N}_{ij}^{cr}$ for both axial and shear surrogate models on the inputs $(\rho_0, \xi, \gamma)$ only, so 3D inputs
    * measure the performance using test RMSE on interpolation and extrapolation (where interpolation is within the zone of data that I keep, and extrapolation is this missing $(\rho_0, \gamma)$ domain of low aspect ratio, high gamma).
    * identify the best GP model using:
        * commercial GP code and a few different models in linear scale, some with $\rho_0$ some with $\rho_0^*$
        * commercial GP code on the log-transform data, $\rho_0$ one vs. $\rho_0^*$ one
        * my custom kernels in log transform (maybe few different versions)
    * use hyperparameter optimization for each of them
    * then make a table with columns "GP model / kernel (version, commercial vs. hand-coded), affine transform (yes/no), log transform (yes/no), test interpolation RMSE, test extrapolation RMSE
        * then hopefully my kernels are the best at extrapolation by far and I can make an argument to use them then..
    * then use the best GP model for the wingbox BC case and adding an extra term for the transverse shear effects
        * and include a nice discussion of this in the paper.