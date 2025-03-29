import ml_buckling as mlb
import numpy as np
import sys
from scipy.optimize import fsolve

# local src for this case
sys.path.append("../src/")
from buckling_analysis import get_buckling_load 

def get_metal_buckling_load(
    comm,
    rho0:float,
    gamma:float,
    num_stiff:int,
    sigma_eig:float,
    stiff_AR:float=20.0,
    ant_hs_ind:int=0,
    b:float=1.0,
    plate_slenderness:float=100,
    is_axial:bool=True,
):
    E = 138e9; nu = 0.3
    G = E / 2.0 / (1 + nu)
    plate_material = mlb.CompositeMaterial(
        E11=E, E22=E, G12=G, nu12=nu,
        ply_angles=[0], ply_fractions=[1.0], ref_axis=[1, 0, 0],
    )

    eig_CF, eig_FEA, stiff_analysis, new_eig_dict, dt = \
    get_buckling_load(
        comm,
        rho0=rho0,
        gamma=gamma,
        plate_slenderness=plate_slenderness,
        num_stiff=num_stiff,
        plate_material=plate_material,
        nelems=2000, #2000
        stiff_AR=stiff_AR,
        sigma_eig=sigma_eig,
        prev_eig_dict=None, #prev_eig_dict,
        is_axial=is_axial,
        solve_buckling=True,
        ant_hs_ind=ant_hs_ind,
        b=b,
        debug=True,
        plot_debug=False,
    )

    return eig_CF, eig_FEA

def axial_load_old(
    comm,
    rho0, 
    gamma, 
    nstiff, 
    prev_dict=None, 
    solve_buckling=True, 
):

    # # iterate on the number of stiffeners until we find a global mode eigenvalue
    # for nstiff in range(1, 6+1):
    stiff_AR = 15.0
    plate_SR = 100.0  # 100.0
    b = 1.0
    h = b / plate_SR  # 10 mm
    nu = 0.3
    E = 138e9
    G = E / 2.0 / (1 + nu)
    # h_w = 0.08 #0.08
    # t_w = h_w / stiff_AR # 0.005
    _nstiff = nstiff if gamma > 0 else 0

    plate_material = mlb.CompositeMaterial(
        E11=E,  # Pa
        E22=E,
        G12=G,
        nu12=nu,
        ply_angles=[0],
        ply_fractions=[1.0],
        ref_axis=[1, 0, 0],
    )

    stiff_material = plate_material

    def gamma_rho0_resid(x):
        h_w = x[0]
        AR = x[1]
        t_w = h_w / stiff_AR

        a = AR * b

        geometry = mlb.StiffenedPlateGeometry(
            a=a, b=b, h=h, num_stiff=_nstiff, h_w=h_w, t_w=t_w
        )
        stiff_analysis = mlb.StiffenedPlateAnalysis(
            comm=comm,
            geometry=geometry,
            stiffener_material=stiff_material,
            plate_material=plate_material,
        )

        return [rho0 - stiff_analysis.affine_aspect_ratio, gamma - stiff_analysis.gamma]

    xopt = fsolve(func=gamma_rho0_resid, x0=(0.08 * gamma / 11.25, rho0))

    h_w = xopt[0]
    AR = xopt[1]
    a = b * AR

    # make a new plate geometry
    geometry = mlb.StiffenedPlateGeometry(
        a=a, b=b, h=h, num_stiff=_nstiff, h_w=h_w, t_w=h_w / stiff_AR
    )
    stiff_analysis = mlb.StiffenedPlateAnalysis(
        comm=comm,
        geometry=geometry,
        stiffener_material=stiff_material,
        plate_material=plate_material,
    )

    _nelems = 2000
    # MIN_Y = 20 / geometry.num_local
    MIN_Y = 5
    MIN_Z = 5  # 5
    N = geometry.num_local
    AR_s = geometry.a / geometry.h_w
    # print(f"AR = {AR}, AR_s = {AR_s}")
    nx = np.ceil(np.sqrt(_nelems / (1.0 / AR + (N - 1) / AR_s)))
    den = 1.0 / AR + (N - 1) * 1.0 / AR_s
    ny = max([np.ceil(nx / AR / N), MIN_Y])
    # nz = max([np.ceil(nx / AR_s), MIN_Z])
    nz = 3

    # my_list = [np.ceil(nx / AR / N), MIN_Y]
    # print(f"{my_list=}")

    # print(f"{nx=} {ny=} {nz=}")

    if solve_buckling:
        stiff_analysis.pre_analysis(
            nx_plate=int(nx),  # 90
            ny_plate=int(ny),  # 30
            nz_stiff=int(nz),  # 5
            nx_stiff_mult=3,
            exx=stiff_analysis.affine_exx,
            exy=0.0,
            clamped=False,
            # _make_rbe=args.rbe,
            _make_rbe=False,
            _explicit_poisson_exp=True,
        )

    comm.Barrier()

    # if comm.rank == 0 and solve_buckling:
    #     print(stiff_analysis)
    # exit()

    if solve_buckling:

        tacs_eigvals, errors = stiff_analysis.run_buckling_analysis(
            sigma=10.0, num_eig=50, write_soln=True  # 50, 100
        )

        # if args.static:
        # stiff_analysis.run_static_analysis(write_soln=True)
        stiff_analysis.post_analysis()

        mode = None

        global_lambda_star = None
        rho0 = stiff_analysis.affine_aspect_ratio
        gamma = stiff_analysis.gamma
        rho0_star = rho0 / (1.0 + gamma)**0.25
        
        
        # only do rho0^* < 1.5 the mode tracking since then we are in the regime of (1,1) mode distortion
        if prev_dict is not None and prev_dict["shape"] == 1:
            if comm.rank == 0:
                global_lambda_star = stiff_analysis.get_matching_global_mode(
                    prev_dict["nondimX"],
                    prev_dict["phi"],
                    min_similarity=0.6
                )

                if global_lambda_star is not None:
                    is_global = stiff_analysis.is_global_mode(
                        imode=stiff_analysis.min_global_mode_index,
                        local_mode_tol=0.75
                    )
                    # if not(is_global):
                    #     global_lambda_star = None
                    #     print(f"mode {imode} was tracked but it is a local mode..")
            
        else: # just use heuristic
            global_lambda_star = stiff_analysis.get_mac_global_mode(
                axial=True,
                min_similarity=0.7,  # 0.5
                local_mode_tol=0.7,
            )

            is_global = global_lambda_star is not None

        # if global_lambda_star is not None:
        #     if np.abs(errors[stiff_analysis.min_global_mode_index]) > 1e-8:
        #         global_lambda_star = None

        mode = "global" if is_global else "local"

        # global_lambda_star = stiff_analysis.min_global_mode_eigenvalue

        if comm.rank == 0:
            stiff_analysis.print_mode_classification()
            print(stiff_analysis)

    else:
        global_lambda_star = None

    # predict the actual eigenvalue
    pred_lambda, mode_type = stiff_analysis.predict_crit_load(axial=True)

    # timoshenko isotropic closed-form
    beta = geometry.a / geometry.b
    TS_gamma = stiff_analysis.gamma / 2.0
    TS_delta = stiff_analysis.delta / 2.0
    timosh_crit = 1e10
    for m in range(1, 100):
        # in book we assume m = 1
        temp = (
            ((1.0 + beta ** 2 / m ** 2) ** 2 + 2.0 * TS_gamma)
            * m ** 2
            / beta ** 2
            / (1.0 + 2.0 * TS_delta)
        )
        if temp < timosh_crit:
            timosh_crit = temp


    # adjust the values for different xi plate
    # had another script that tried to adaptively select ply_angle to achieve certain xi,gamma,rho0 but scipy.optimize.fsolve
    # kept not converging..
    
    # first adjust N11_CF by xi diff
    pred_lambda += 2*(1.0 - stiff_analysis.xi_plate)
    # second adjust FEA by xi diff
    if solve_buckling and global_lambda_star is not None:
        global_lambda_star += 2*(1.0 - stiff_analysis.xi_plate)

    # min_eigval = tacs_eigvals[0]
    # rel_err = (pred_lambda - global_lambda_star) / pred_lambda
    if comm.rank == 0 and solve_buckling:
        print(f"{stiff_analysis.intended_Nxx}")

        if global_lambda_star is not None:
            global_lambda_star = global_lambda_star.real

        print(f"Mode type predicted as {mode_type}")
        print(f"\ttimoshenko CF lambda = {timosh_crit}")
        print(f"\tmy CF min lambda = {pred_lambda}")
        print(f"\tFEA min lambda = {global_lambda_star}")

        # # write the data to csv  after each buckling solve
        # my_df_dict = {
        #     "nstiff" : [nstiff],
        #     "mode" : [mode],
        #     "rho0": [rho0],
        #     "gamma": [gamma],
        #     "eig_CF": [pred_lambda],
        #     "eig_FEA": [global_lambda_star],
        # }
        # my_df = pd.DataFrame(my_df_dict)
        # my_df.to_csv("7-nstiff-compare.csv", header=first, mode="w" if first else "a")

    comm.Barrier()

    eig_dict = None
    if solve_buckling and comm.rank == 0:
        eig_dict = {
            "nondimX" : stiff_analysis.nondim_X,
            "phi" : stiff_analysis.min_global_eigmode,
            "shape" : stiff_analysis._min_global_mode_shape
        }

    if global_lambda_star is not None:
        global_lambda_star = global_lambda_star.real

    return [global_lambda_star, pred_lambda, timosh_crit, eig_dict]