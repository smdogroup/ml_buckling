# V1: 0.087 axial, 0.5 shear (smoothed or not)
    # best extrap (current baseline)
    # th = np.array([1, 8, 4, 1, 1, 0.1])

    # axial, log optimal - from k-fold optimization
    th = np.array([1.18633894, 4.07496564, 8.54108165, 0.1       , 1.27521468,
       0.1])
    
    # temp change check
    th[3] = 1.0

    # optimized when relu_alph = 1.0 is fixed
    # th = np.array([1.35287848, 4.0041452 , 8.36474745, 1.        , 0.9702865 ,
    #    0.1       ])
    th = np.array([1.36363751, 4.23877129, 8.53337483, 1.        , 1.11124537,
       0.1       , 9.38337746])
    # th = np.array([ 1.16506573,  4.82146699,  1.01831129,  5.        ,  1.0172672 ,
    #     0.1       , 10.        ])

    th = np.array([1.16501839, 4.82083506, 1.01800903, 5.        , 1.13204931,
       0.1       , 8.11339131])
    th[-2] = 0.01

    # latest version, however needs th[-2] = 1e-2 not this value to extrapolate well
    th = np.array([6.40151232e-01, 3.38465103e+00, 6.02157464e-01, 1.00000000e+01,
       9.98604621e-01, 1.90107832e-03, 7.39807708e-01])
    
    # th[3] = 5.0
    th[-2] = 1e-2
    # th[-2] = 0.0

    # optimized with th[-2] lb of 1e-2 this time
    th = np.array([0.70681975, 4.00450695, 0.1       , 9.24252847, 0.99095599,
       0.01      , 3.15286551])
    
    th = np.array([0.47032850570140605, 3.452939774497288, 0.10290088334566738, 10.0, 1.004123377743182, 0.015069986442508518, 3.0841005316039336])



    # best one:
    # latest version, however needs th[-2] = 1e-2 not this value to extrapolate well
    th = np.array([6.40151232e-01, 3.38465103e+00, 6.02157464e-01, 1.00000000e+01,
       9.98604621e-01, 1.90107832e-03, 7.39807708e-01])
    th[-2] = 1e-2