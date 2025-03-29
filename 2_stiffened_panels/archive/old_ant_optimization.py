# first solve with current bounds
    # bounds1 = [(-3, 1), (-3, ub_log_rho)]
    # initial_guess1 = guess_x0
    # # sol = root(gamma_rho0_resid, initial_guess, method='trust-constr', bounds=bounds)
    # sol1 = minimize(objective, initial_guess1, bounds=bounds1, method='trust-constr')
    # xopt1 = sol.x
    # myresid1 = gamma_rho0_resid(xopt1)
    # solved1 = np.linalg.norm(myresid1) < 1e-5

    # # homotopic residual now?

    # # then solve again with bounds reduced
    # initial_guess1 = guess_x0
    # # sol = root(gamma_rho0_resid, initial_guess, method='trust-constr', bounds=bounds)
    # sol1 = minimize(objective, initial_guess1, bounds=bounds1, method='trust-constr')
    # xopt1 = sol.x
    # myresid1 = gamma_rho0_resid(xopt1)
    # solved1 = np.linalg.norm(myresid1) < 1e-5

    # ct = 0
    # solved = False
    # while not(solved) and ct < 5:
    #     ct += 1
    #     # xopt = sopt.fsolve(func=gamma_rho0_resid, x0=guess_x0, xtol=1e-8)
    #     bounds = [(-3, 1), (-3, ub_log_rho)]
    #     initial_guess = guess_x0
    #     # sol = root(gamma_rho0_resid, initial_guess, method='trust-constr', bounds=bounds)
    #     sol = minimize(objective, initial_guess, bounds=bounds, method='trust-constr')
    #     xopt = sol.x
    #     myresid = gamma_rho0_resid(xopt)
    #     solved = np.linalg.norm(myresid) < 1e-5


# temporary debug change to metal
    # if debug:
    #     change_to_metal = False
    #     if change_to_metal:
    #         nu = 0.3
    #         E = 138e9
    #         G = E / 2.0 / (1 + nu)
    #         plate_material = mlb.CompositeMaterial(
    #             E11=E,  # Pa
    #             E22=E,
    #             G12=G,
    #             nu12=nu,
    #             ply_angles=[0],
    #             ply_fractions=[1.0],
    #             ref_axis=[1, 0, 0],
    #         )
    #     else:
    #         ply_angle = 30.0

    #         plate_material = mlb.CompositeMaterial.solvay5320(
    #             ply_angles=[ply_angle],
    #             ply_fractions=[1.0],
    #             ref_axis=[1, 0, 0],
    #         )
        

    #     rho0 = 5.0
    #     gamma = 3.0
    #     plate_slenderness = 100.0
    #     nstiff = 5

    #     # # change to zero stiffeners
    #     # gamma = 0.0
    #     # nstiff = 0