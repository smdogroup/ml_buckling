import time
import ml_buckling as mlb
import numpy as np
from ant_optimization import ant_optimization

def get_buckling_load(
    comm,
    rho0:float, 
    gamma:float,
    plate_slenderness:float,
    num_stiff:int,
    plate_material,
    nelems:int=2000,
    stiff_AR:float=20.0,
    sigma_eig:float=5.0,
    prev_eig_dict:dict=None,
    is_axial:bool=True, 
    solve_buckling:bool=True, 
    ant_hs_ind:int=0,
    b:float=1.0,
    debug:bool=False,
    plot_debug:bool=False,
):
    
    start_time = time.time()

    # set fixed geometry parameters
    b = 1.0
    h = b / plate_slenderness
    # h = 0.1
    # b = h * plate_slenderness
    nstiff = num_stiff if gamma > 0 else 0

    stiff_material = plate_material


    # ant optimization method here to minimize R(xhat)
    # ------------------------------------------------

    h_w, AR, a = ant_optimization(
        comm,
        plate_material,
        stiff_material,
        nstiff,
        stiff_AR,
        b,
        h,
        rho0,
        gamma,
        debug,
        plot_debug=plot_debug,
        hs_ind=ant_hs_ind,
    )

    print(f"{comm.rank} checkpt-2", flush=True)
    

    # make a new plate geometry
    # -------------------------

    geometry = mlb.StiffenedPlateGeometry(
        a=a, b=b, h=h, num_stiff=nstiff, h_w=h_w, t_w=h_w / stiff_AR
    )
    stiff_analysis = mlb.StiffenedPlateAnalysis(
        comm=comm,
        geometry=geometry,
        stiffener_material=stiff_material,
        plate_material=plate_material,
    )

    print(f"{comm.rank} checkpt-1", flush=True)

    #if comm.rank == 0:
    print(f"achieved {stiff_analysis.gamma=} {stiff_analysis.affine_aspect_ratio=}", flush=True)
    # exit()

    _nelems = nelems
    MIN_Y = 5
    MIN_Z = 5  # 5
    N = geometry.num_local
    AR_s = geometry.a / geometry.h_w
    # den = (1.0 / AR + (N - 1) / AR_s)
    # print(f"{AR_s=} {den=}")
    nx = np.ceil(np.sqrt(_nelems / (1.0 / AR + (N - 1) / AR_s)))
    print(f"{nx=}")
    nx = max([nx, 5])
    den = 1.0 / AR + (N - 1) * 1.0 / AR_s
    ny = max([np.ceil(nx / AR / N), MIN_Y])
    nz = 3

    # print(f"{nx=} {ny=} {nz=}")
    print(f"{comm.rank} checkpt0", flush=True)

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

    print(f"{comm.rank} checkpt1", flush=True)
    comm.Barrier()

    if debug:
        # run a linear static analysis to debug
        stiff_analysis.run_static_analysis(base_path=None, write_soln=True)

    if solve_buckling:

        tacs_eigvals, errors = stiff_analysis.run_buckling_analysis(
            sigma=sigma_eig, num_eig=100, write_soln=True  # 50, 100
        )
        print(f"{comm.rank} checkpt2", flush=True)
        stiff_analysis.post_analysis()
        print(f"{comm.rank} checkpt3", flush=True)

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
            if comm.rank == 0:
                eig_FEA = stiff_analysis.get_mac_global_mode(
                    axial=is_axial,
                    min_similarity=0.75,  # 0.5
                    local_mode_tol=0.75, #0.8
                )

        comm.Barrier()

        if eig_FEA is not None and comm.rank == 0:
            if np.abs(errors[stiff_analysis.min_global_mode_index]) > 1e-8:
                eig_FEA = None

        print(f"{comm.rank=} pre-broadcast")
        eig_FEA = comm.bcast(eig_FEA, root=0)
        print(f"{comm.rank=} {eig_FEA=}")

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
