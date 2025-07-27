# Comparison / Verification of Single vs Multi Ply Buckling Loads for the ML buckling journal paper

* goal is to show that the buckling loads are the same as long as the nondim parameters are the same between single and multi-ply
laminate cases

## Case 1 - Lightly stiffened, AR = 1

* note, we go by final ND buckling load here..
* baseline multiply settings for small stiffening: _h_w, _AR = 0.02, 1.0
* also plate SR = 100 here (so a bit thicker)
* identified the hw_mult, AR_mult here of hw_mult, AR_mult = 0.85, 1.3
* almost an exact match within 0.22% error when xi difference is corrected for
* hw_mult and AR_mult needed since mix of 0 and 90 plies results in low xi but higher gamma and lower rho0 (so do need to adjust to single ply)

single ply: theta = 0.0
    single ply, eigval0 = 4.081e+00
        rho0=6.962e-01, xi=3.814e-01
        gamma=6.990e-02, zeta=2.481e-03, delta=2.890e-03
        buckle_Nxx_nd=4.081e+00

multi ply: [0/90]_s laminate
    multi ply, eigval0 = 2.650e+00
        rho0=6.868e-01, xi=2.620e-01
        gamma=7.266e-02, zeta=1.343e-03, delta=4.000e-03
        buckle_Nxx_nd=3.834e+00


## Case 2 - More stiff, and less slender

* changed to [0/65]_s sym laminate with equal ply fractions to make xi match better..

single ply, eigval0 = 6.032e+00
        rho0=1.370e+00, xi=3.810e-01
        gamma=1.263e+00, zeta=2.757e-04, delta=9.083e-03
        buckle_Nxx_nd=6.032e+00

multi ply, eigval0 = 4.174e+00
        rho0=1.306e+00, xi=3.599e-01
        gamma=1.280e+00, zeta=1.351e-04, delta=1.200e-02
        buckle_Nxx_nd=6.411e+00

* dropped plate slenderness by sqrt(2) => so zeta values match (now really good agreement!)
multi ply, eigval0 = 4.030e+00
        rho0=1.353e+00, xi=3.802e-01
        gamma=1.227e+00, zeta=2.566e-04, delta=1.379e-02
        buckle_Nxx_nd=6.058e+00


## Case 3 - even more stiff, gamma = 4.0

single ply, eigval0 = 1.077e+01
        rho0=1.366e+00, xi=3.784e-01
        gamma=4.407e+00, zeta=2.757e-04, delta=1.780e-02
        buckle_Nxx_nd=1.077e+01

multi ply, eigval0 = 7.264e+00
        rho0=1.346e+00, xi=3.762e-01
        gamma=4.383e+00, zeta=2.566e-04, delta=2.749e-02
        buckle_Nxx_nd=1.099e+01

## yeah looks like they are very close so far
## Case 4 - higher xi values (more in-plane shear effects)

* theta = 30.0 deg in single ply
* [0/45/-45/90]_s laminate with [1,2,2,1]_s relative ply fractions => xi = 0.65 now
* 3% error here despite very different laminates! but they have very similar nondim params..

single ply, eigval0 = 9.266e+00
        rho0=1.621e+00, xi=6.605e-01
        gamma=3.671e+00, zeta=1.043e-04, delta=1.655e-02
        buckle_Nxx_nd=9.266e+00

multi ply, eigval0 = 1.382e+01
        rho0=1.588e+00, xi=6.494e-01
        gamma=3.630e+00, zeta=9.470e-05, delta=2.749e-02
        buckle_Nxx_nd=9.530e+00