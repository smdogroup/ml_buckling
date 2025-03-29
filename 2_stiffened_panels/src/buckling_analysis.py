import time
import ml_buckling as mlb
import scipy.optimize as sopt
import numpy as np

def get_buckling_load(
    comm,
    rho0:float, 
    gamma:float,
    plate_slenderness:float,
    num_stiff:int,
    plate_material,
    nelems:int=2000,
    prev_eig_dict:dict=None,
    is_axial:bool=True, 
    solve_buckling:bool=True, 
    debug:bool=False,
):

    start_time = time.time()

    stiff_AR = 20.0
    # stiff_AR = 10.0
    # plate_SR = 100.0  # 100.0
    h = 0.1
    # h = 1.0 # need larger h for more accurate buckling analysis potentially
    b = h * plate_slenderness
    # h = b / plate_slenderness  # 10 mm
    nstiff = num_stiff if gamma > 0 else 0

    # temporary debug change to metal
    if debug:
        change_to_metal = False
        if change_to_metal:
            nu = 0.3
            E = 138e9
            G = E / 2.0 / (1 + nu)
            plate_material = mlb.CompositeMaterial(
                E11=E,  # Pa
                E22=E,
                G12=G,
                nu12=nu,
                ply_angles=[0],
                ply_fractions=[1.0],
                ref_axis=[1, 0, 0],
            )
        else:
            ply_angle = 30.0

            plate_material = mlb.CompositeMaterial.solvay5320(
                ply_angles=[ply_angle],
                ply_fractions=[1.0],
                ref_axis=[1, 0, 0],
            )
        

        rho0 = 5.0
        gamma = 3.0
        plate_slenderness = 100.0
        nstiff = 5

        # # change to zero stiffeners
        # gamma = 0.0
        # nstiff = 0

    stiff_material = plate_material

    target_log_gamma = np.log10(1.0+gamma)
    target_log_rho = np.log10(rho0)

    if nstiff > 0:
        def gamma_rho0_resid(x):
            # x is [log10(h_w), log10(AR)]
            h_w = 10**x[0]
            AR = 10**x[1]
            t_w = h_w / stiff_AR

            a = AR * b

            geometry = mlb.StiffenedPlateGeometry(
                a=a, b=b, h=h, num_stiff=nstiff, h_w=h_w, t_w=t_w
            )
            stiff_analysis = mlb.StiffenedPlateAnalysis(
                comm=comm,
                geometry=geometry,
                stiffener_material=stiff_material,
                plate_material=plate_material,
            )

            pred_log_rho0 = np.log10(stiff_analysis.affine_aspect_ratio)
            pred_log_gamma = np.log10(1.0 + stiff_analysis.gamma)
            return [target_log_rho - pred_log_rho0, target_log_gamma - pred_log_gamma]
    else: # nstiff == 0
        def gamma_rho0_resid(x):
            # x is [log10(h_w), log10(AR)]
            h_w = 10**x[0]
            AR = 10**x[1]
            t_w = h_w / stiff_AR

            a = AR * b

            geometry = mlb.StiffenedPlateGeometry(
                a=a, b=b, h=h, num_stiff=nstiff, h_w=h_w, t_w=t_w
            )
            stiff_analysis = mlb.StiffenedPlateAnalysis(
                comm=comm,
                geometry=geometry,
                stiffener_material=stiff_material,
                plate_material=plate_material,
            )

            pred_log_rho0 = np.log10(stiff_analysis.affine_aspect_ratio)
            pred_log_gamma = np.log10(1.0 + stiff_analysis.gamma)
            return [target_log_rho - pred_log_rho0, np.log10(h_w) + 7]
    
    # get best initial condition 
    min_err = np.inf
    for log_hw in np.linspace(-3, 1, 20):
        for log_rho in np.linspace(-2, 2, 20):
            myresid = gamma_rho0_resid([log_hw, log_rho])
            resid_norm = np.linalg.norm(np.array(myresid))
            if resid_norm < min_err:
                if nstiff == 0:
                    guess_x0 = (-7, log_rho)
                else:
                    guess_x0 = (log_hw, log_rho)
                min_err = resid_norm
    print(f"{guess_x0=} {resid_norm=}")

    ct = 0
    solved = False
    while not(solved) and ct < 5:
        ct += 1
        xopt = sopt.fsolve(func=gamma_rho0_resid, x0=guess_x0, xtol=1e-8)
        myresid = gamma_rho0_resid(xopt)
        solved = np.linalg.norm(myresid) < 1e-5

    h_w = 10**(xopt[0])
    AR = 10**(xopt[1])
    a = b * AR

    # # now double check residual error
    # resid = gamma_rho0_resid(xopt)
    # print(f"{xopt=}")
    # print(f"{resid=}")

    # make a new plate geometry
    geometry = mlb.StiffenedPlateGeometry(
        a=a, b=b, h=h, num_stiff=nstiff, h_w=h_w, t_w=h_w / stiff_AR
    )
    stiff_analysis = mlb.StiffenedPlateAnalysis(
        comm=comm,
        geometry=geometry,
        stiffener_material=stiff_material,
        plate_material=plate_material,
    )

    # print(f"achieved {stiff_analysis.gamma=} {stiff_analysis.affine_aspect_ratio=}")
    # exit()

    _nelems = nelems
    MIN_Y = 5
    MIN_Z = 5  # 5
    N = geometry.num_local
    AR_s = geometry.a / geometry.h_w
    nx = np.ceil(np.sqrt(_nelems / (1.0 / AR + (N - 1) / AR_s)))
    den = 1.0 / AR + (N - 1) * 1.0 / AR_s
    ny = max([np.ceil(nx / AR / N), MIN_Y])
    nz = 3

    # print(f"{nx=} {ny=} {nz=}")

    if solve_buckling:
        stiff_analysis.pre_analysis(
            nx_plate=int(nx),  # 90
            ny_plate=int(ny),  # 30
            nz_stiff=int(nz),  # 5
            nx_stiff_mult=2,
            exx=stiff_analysis.affine_exx if is_axial else 0.0,
            exy=stiff_analysis.affine_exy if not(is_axial) else 0.0,
            clamped=False,
            _make_rbe=False,
            _explicit_poisson_exp=True,
        )

    comm.Barrier()

    if debug:
        # run a linear static analysis to debug
        stiff_analysis.run_static_analysis(base_path=None, write_soln=True)

    if solve_buckling:

        tacs_eigvals, errors = stiff_analysis.run_buckling_analysis(
            sigma=5.0, num_eig=100, write_soln=True  # 50, 100
        )
        stiff_analysis.post_analysis()

        eig_FEA = None
        rho0 = stiff_analysis.affine_aspect_ratio
        gamma = stiff_analysis.gamma
        rho0_star = rho0 / (1.0 + gamma)**0.25
        
        # only do rho0^* < 1.5 the mode tracking since then we are in the regime of (1,1) mode distortion
        # should be less than 1.5 but this is more conservative
        if rho0_star < 1.5 and (prev_eig_dict is not None and prev_eig_dict["shape"] == 1): # ensure (1,1) shape before start mode tracking
            if comm.rank == 0:
                # print(f"{prev_dict=}")
                eig_FEA = stiff_analysis.get_matching_global_mode(
                    prev_eig_dict["nondimX"],
                    prev_eig_dict["phi"],
                    min_similarity=0.7,
                )

                if eig_FEA is not None:
                    imode = stiff_analysis.min_global_mode_index
                    is_global = stiff_analysis.is_global_mode(
                        imode=stiff_analysis.min_global_mode_index,
                        local_mode_tol=0.75
                    )
                    if not(is_global):
                        eig_FEA = None
                        print(f"mode {imode} was tracked but it is a local mode..")
            
        else: # just use heuristic
            eig_FEA = stiff_analysis.get_mac_global_mode(
                axial=is_axial,
                min_similarity=0.75,  # 0.5
                local_mode_tol=0.75, #0.8
            )

        if eig_FEA is not None:
            if np.abs(errors[stiff_analysis.min_global_mode_index]) > 1e-8:
                eig_FEA = None

        if comm.rank == 0:
            stiff_analysis.print_mode_classification()
            print(stiff_analysis)

    else:
        eig_FEA = None

    # predict the actual eigenvalue
    eig_CF, mode_type = stiff_analysis.predict_crit_load(axial=is_axial)

    # min_eigval = tacs_eigvals[0]
    # rel_err = (pred_lambda - global_lambda_star) / pred_lambda
    if comm.rank == 0 and solve_buckling:
        print(f"{stiff_analysis.intended_Nxx}")

        print(f"Mode type predicted as {mode_type}")
        print(f"\t{eig_CF=}")
        print(f"\t{eig_FEA=}")

    comm.Barrier()

    eig_dict = None
    if solve_buckling and comm.rank == 0:
        eig_dict = {
            "nondimX" : stiff_analysis.nondim_X,
            "phi" : stiff_analysis.min_global_eigmode,
            "shape" : stiff_analysis._min_global_mode_shape
        }

    dt = time.time() - start_time

    # returns (CF_eig, FEA_eig) as follows:
    return eig_CF, eig_FEA, stiff_analysis, eig_dict, dt