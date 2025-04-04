from funtofem import *
import numpy as np
from mpi4py import MPI
from tacs import pytacs
import os

def make_model_and_solver(
    comm,
    args,
    model_name:str,
):
    
    # ML or CF callback and funtofem model
    # ------------------------------------
    
    if args.useML:
        from __gp_callback import gp_callback_generator
    else:
        from __closed_form_callback import closed_form_callback as callback
        
    f2f_model = FUNtoFEMmodel(model_name)


    # bodies and struct DVs
    # ---------------------
    
    wing = Body.aeroelastic("wing", boundary=2)

    # make the component groups list for TACS
    nribs = 23
    nOML = nribs - 1
    component_groups = [f"rib{irib}" for irib in range(1, nribs + 1)]
    for prefix in ["spLE", "spTE", "uOML", "lOML"]:
        component_groups += [f"{prefix}{iOML}" for iOML in range(1, nOML + 1)]
    component_groups = sorted(component_groups)

    if args.useML:
        # use component groups to make gp callback so that one PanelGPs object per TACS component
        callback = gp_callback_generator(component_groups)

    # make each struct design variable
    for icomp, comp in enumerate(component_groups):
        Variable.structural(
            f"{comp}-" + TacsSteadyInterface.LENGTH_VAR, value=0.04
        ).set_bounds(
            lower=0.0,
            scale=1.0,
            state=True,  # need the length & width to be state variables
        ).register_to(
            wing
        )

        struct_active = True

        # stiffener pitch variable
        Variable.structural(f"{comp}-spitch", value=0.20).set_bounds(
            lower=0.05, upper=0.5, scale=1.0, active=struct_active
        ).register_to(wing)

        # panel thickness variable, shortened DV name for ESP/CAPS, nastran requirement here
        Variable.structural(f"{comp}-T", value=0.02).set_bounds(
            lower=0.002, upper=0.1, scale=100.0, active=struct_active
        ).register_to(wing)

        # stiffener height
        Variable.structural(f"{comp}-sheight", value=0.05).set_bounds(
            lower=0.0254 / 0.8, upper=0.1, scale=10.0, active=struct_active
        ).register_to(wing)

        # stiffener thickness
        Variable.structural(f"{comp}-sthick", value=0.02).set_bounds(
            lower=0.002, upper=0.1, scale=100.0, active=struct_active
        ).register_to(wing)

        Variable.structural(
            f"{comp}-" + TacsSteadyInterface.WIDTH_VAR, value=0.02
        ).set_bounds(
            lower=0.0,
            scale=1.0,
            state=True,  # need the length & width to be state variables
        ).register_to(
            wing
        )

    wing.register_to(f2f_model)

    
    # scenarios
    # ---------

    T_sl = 288.15
    q_sl = 14.8783e3

    # pullup
    pull_up = Scenario.steady(
        "pull_up",
        steps=1,
        coupled=False,
    )
    clift = Function.lift(body=0).register_to(pull_up)
    pull_up_ks = (
        Function.ksfailure(ks_weight=100, safety_factor=1.5)
        .optimize(scale=1.0, upper=1.0, objective=False, plot=True, plot_name="ks-pullup")
        .register_to(pull_up)
    )
    mass_wingbox = (
        Function.mass()
        .optimize(scale=1.0e-3, objective=True, plot=True, plot_name="mass")
        .register_to(pull_up)
    )

    qfactor = 1.0
    aoa_pull_up = pull_up.get_variable("AOA").set_bounds(lower=6.0, value=7.0, upper=14.0, scale=10)
    pull_up.set_temperature(T_ref=T_sl, T_inf=T_sl)
    pull_up.set_flow_ref_vals(qinf=qfactor*q_sl)
    pull_up.register_to(f2f_model)

    # pushdown
    push_down = Scenario.steady(
        "push_down",
        steps=1,
        coupled=False,
    )
    clift_pd = Function.lift(body=0).register_to(push_down)
    push_down_ks = (
        Function.ksfailure(ks_weight=args.ksWeight, safety_factor=1.5)
        .optimize(scale=1.0, upper=1.0, objective=False, plot=True, plot_name="ks-pushdown")
        .register_to(push_down)
    )
    Function.mass().register_to(push_down)

    aoa_push_down = push_down.get_variable("AOA").set_bounds(lower=-10.0, value=-6.0, upper=-2.0, scale=10)
    push_down.set_temperature(T_ref=T_sl, T_inf=T_sl)
    push_down.set_flow_ref_vals(qinf=qfactor*q_sl)
    push_down.register_to(f2f_model)

    # load factor constraints (composite functions)
    # ---------------------------------------------

    # compute est. weight of the vehicle
    mass_wing = 10.147 * mass_wingbox**0.8162  # Elham regression model
    mass_payload = 14.5e3  # kg
    mass_frame = 25e3  # kg
    mass_fuel_res = 2e3  # kg
    LGM = mass_payload + mass_frame + mass_fuel_res + 2 * mass_wing
    LGW = 9.81 * LGM  # kg => N

    # pull up load factor constraint
    pull_up_lift = clift * 2 * q_sl * 45.5 
    pull_up_LF = pull_up_lift - 2.5 * LGW
    pull_up_LF.set_name("pull_up_LF").optimize(
        lower=0.0, upper=0.0, scale=1e-4, objective=False, plot=True
    ).register_to(f2f_model)

    # push down load factor constraint
    push_down_lift = clift_pd * 2 * q_sl * 45.5 
    push_down_LF = push_down_lift + 1.0 * LGW # -1.0 load factor
    push_down_LF.set_name("push_down_LF").optimize(
        lower=0.0, upper=0.0, scale=1e-4, objective=False, plot=True
    ).register_to(f2f_model)

    # struct adjacency composite functions
    # ------------------------------------
    
    thick_adj = 2.5e-3

    comp_groups = ["spLE", "spTE", "uOML", "lOML"]
    comp_nums = [nOML for i in range(len(comp_groups))]
    adj_types = ["T"]
    adj_types += ["sthick", "sheight"]
    adj_values = [thick_adj, thick_adj, 10e-3]
    for igroup, comp_group in enumerate(comp_groups):
        comp_num = comp_nums[igroup]
        for icomp in range(1, comp_num):
            # no constraints across sob (higher stress there)
            for iadj, adj_type in enumerate(adj_types):
                adj_value = adj_values[iadj]
                name = f"{comp_group}{icomp}-{adj_type}"
                # print(f"name = {name}", flush=True)
                left_var = f2f_model.get_variables(f"{comp_group}{icomp}-{adj_type}")
                right_var = f2f_model.get_variables(f"{comp_group}{icomp+1}-{adj_type}")
                # print(f"left var = {left_var}, right var = {right_var}")
                adj_constr = left_var - right_var
                adj_constr.set_name(f"{comp_group}{icomp}-adj_{adj_type}").optimize(
                    lower=-adj_value, upper=adj_value, scale=10.0, objective=False
                ).register_to(f2f_model)

        for icomp in range(1, comp_num + 1):
            skin_var = f2f_model.get_variables(f"{comp_group}{icomp}-T")
            sthick_var = f2f_model.get_variables(f"{comp_group}{icomp}-sthick")
            sheight_var = f2f_model.get_variables(f"{comp_group}{icomp}-sheight")
            spitch_var = f2f_model.get_variables(f"{comp_group}{icomp}-spitch")

            # stiffener - skin thickness adjacency here
            adj_value = thick_adj
            adj_constr = 15.0 * skin_var - sthick_var
            adj_constr.set_name(f"{comp_group}{icomp}-skin_stiff_T").optimize(
                lower=0.0, scale=10.0, objective=False
            ).register_to(f2f_model)

            # minimum stiffener spacing pitch - sheight
            min_spacing_constr = spitch_var - sheight_var
            min_spacing_constr.set_name(f"{comp_group}{icomp}-sspacing").optimize(
                lower=0.0, scale=1.0, objective=False
            ).register_to(f2f_model)

            # minimum stiffener AR
            min_stiff_AR = sheight_var - 5.0 * sthick_var
            min_stiff_AR.set_name(f"{comp_group}{icomp}-minstiffAR").optimize(
                lower=0.0, scale=1.0, objective=False
            ).register_to(f2f_model)

            # maximum stiffener AR (for regions with tensile strains where crippling constraint won't be active)
            max_stiff_AR = sheight_var - 30.0 * sthick_var
            max_stiff_AR.set_name(f"{comp_group}{icomp}-maxstiffAR").optimize(
                upper=0.0, scale=1.0, objective=False
            ).register_to(f2f_model)

    # discipline solvers (just TACS here)
    # -----------------------------------

    solvers = SolverManager(comm)

    output_dir = "design/struct/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    dat_file = "design/struct/tacs.dat"

    # inertial loads and auxElems

    world_rank = comm.Get_rank()
    if world_rank < args.procs:
        color = 1
    else:
        color = MPI.UNDEFINED
    tacs_comm = comm.Split(color, world_rank)

    struct_id = None
    assembler = None

    # make the tacs panel dimensions
    tacs_panel_dimensions = TacsPanelDimensions(
        comm=comm,
        panel_length_dv_index=0,
        panel_width_dv_index=5,
    )
    gen_output = None

    if world_rank < args.procs:
        
        # create TACSInterface manually
        # so that we can add inertial and pressure loads to the wing through
        # tacs auxillary elements
        my_pytacs = pytacs.pyTACS(dat_file, tacs_comm, options={})
        my_pytacs.initialize(callback)
        
        # add auxillary elements for our custom DV-dependent loading (gravity through mass dependence)
        # and the constant pressure loading
        SP = my_pytacs.createStaticProblem("struct-benchmark")
        #compIDs = my_pytacs.selectCompIDs('lOML') # the lower skin
        #SP.addPressureToComponents(compIDs, -30e3) # 30 kPa to lower skin (+press is outward so down on lower skin, -press is +z)
        SP.addInertialLoad([0.0, 0.0, -9.81])
        
        SP._updateAssemblerVars() # this writes auxElems into assembler
        # print(f"{SP.auxElems=}", flush=True)
        # exit()
        
        gen_output = TacsOutputGenerator(output_dir, f5=my_pytacs.outputViewer)
        
        # get the struct ids
        # get list of local node IDs with global size, with -1 for nodes not owned by this proc
        num_nodes = my_pytacs.meshLoader.bdfInfo.nnodes
        bdfNodes = range(num_nodes)
        local_tacs_ids = my_pytacs.meshLoader.getLocalNodeIDsFromGlobal(
            bdfNodes, nastranOrdering=False
        )
        
        # get number of non -1 tacs ids, total number of actual tacs_ids
        n_tacs_ids = len([tacs_id for tacs_id in local_tacs_ids if tacs_id != -1])
        
        # reverse the tacs id to nastran ids map since we want tacs_id => nastran_id - 1
        nastran_ids = np.zeros((n_tacs_ids), dtype=np.int64)
        for nastran_id_m1, tacs_id in enumerate(local_tacs_ids):
            if tacs_id != -1:
                nastran_ids[tacs_id] = int(nastran_id_m1 + 1)
        
        # convert back to list of nastran_ids owned by this processor in order
        struct_id = list(nastran_ids)
    
        tacs_panel_dimensions.panel_length_constr = my_pytacs.createPanelLengthConstraint("PanelLengthCon")
        tacs_panel_dimensions.panel_length_constr.addConstraint(
            TacsSteadyInterface.LENGTH_CONSTR,
            dvIndex=0
        )

        tacs_panel_dimensions.panel_width_constr = my_pytacs.createPanelWidthConstraint("PanelWidthCon")
        tacs_panel_dimensions.panel_width_constr.addConstraint(
            TacsSteadyInterface.WIDTH_CONSTR,
            dvIndex=5
        )

        assembler = SP.assembler
        
    # make the steady tacs interface manually
    solvers.structural = TacsSteadyInterface(
        comm,
        f2f_model,
        assembler,
        gen_output=gen_output if args.struct_output else None,
        thermal_index=6,
        tacs_comm=tacs_comm,
        tacs_panel_dimensions=tacs_panel_dimensions,
        struct_id=struct_id,
    )

    transfer_settings = TransferSettings(
        npts=2000,
        isym=1,
        beta=0.3
    )

    return f2f_model, solvers, transfer_settings
# end of the make_model_and_solver function