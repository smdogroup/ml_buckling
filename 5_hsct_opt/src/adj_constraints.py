def add_adjacency_constraints(
    comm,
    f2f_model,
    opt_manager,
    opt_problem,
):
    
    # skin thickness adjacency constraints
    variables = f2f_model.get_variables()
    adjacency_scale = 10.0
    thick_adj = 2.5e-3

    nribs = 20
    nspars = 40
    nOML = nribs - 1

    adj_types = ["pthick", "sthick", "sheight"]
    adj_values = [2.5e-3, 2.5e-3, 10e-3]

    adj_prefix_lists = []
    ncomp = 0

    # OMLtop, bot chordwise adjacency
    for prefix in ["OMLtop", "OMLbot"]:
        # chordwise adjacency
        adj_prefix_lists += [
            [f"{prefix}{iOML}-{ispar}", f"{prefix}{iOML}-{ispar+1}"]
            for iOML in range(1, nOML + 1)
            for ispar in range(1, nspars + 1)
        ]
        # spanwise adjacency
        adj_prefix_lists += [
            [f"{prefix}{iOML}-{ispar}", f"{prefix}{iOML+1}-{ispar}"]
            for iOML in range(1, nOML)
            for ispar in range(1, nspars + 2)
        ]

    # rib chordwise adjacency
    adj_prefix_lists += [
        [f"rib{irib}-{ispar}", f"rib{irib}-{ispar+1}"]
        for irib in range(1, nribs + 1)
        for ispar in range(1, nspars + 1)
    ]

    # spar adjacencies
    spar_prefixes = [f"spar{ispar}" for ispar in range(1, nspars + 1)] + [
        "LEspar",
        "TEspar",
    ]
    adj_prefix_lists += [
        [f"{spar_prefix}-{iOML}", f"{spar_prefix}-{iOML+1}"]
        for spar_prefix in spar_prefixes
        for iOML in range(1, nOML + 1)
    ]

    # print(f"num adj prefix lists = {len(adj_prefix_lists)}", flush=True)
    # exit()


    def get_var_index(variable):
        for ivar, var in enumerate(f2f_model.get_variables(optim=True)):
            if var == variable:
                return ivar


    n = len(adj_prefix_lists)
    if comm.rank == 0:
        print(f"starting adj functions up to {3*len(adj_prefix_lists)}", flush=True)

    nvariables = len(f2f_model.get_variables(optim=True))
    # nvariables = 6456 # TODO : Fix this later.. [said it was 9684 instead of 6456 when I used len(...)]

    for i, prefix_list in enumerate(adj_prefix_lists):
        left_prefix = prefix_list[0]
        right_prefix = prefix_list[1]
        if comm.rank == 0:
            print(f"adjacency funcs {ncomp}")
            if i < n - 1:
                print("\033[1A", end="")

        for iadj, adj_type in enumerate(adj_types):
            adj_value = adj_values[iadj]
            left_var = f2f_model.get_variables(f"{left_prefix}-{adj_type}")
            right_var = f2f_model.get_variables(f"{right_prefix}-{adj_type}")
            if left_var is not None and right_var is not None:
                adj_constr = left_var - right_var
                adj_constr.set_name(f"{left_prefix}-adj{i}_{adj_type}").optimize(
                    lower=-adj_value, upper=adj_value, scale=10.0, objective=False
                ).setup_sparse_gradient(f2f_model)

                opt_manager.register_sparse_constraint(opt_problem, adj_constr)

                del adj_constr
                ncomp += 1

    if comm.rank == 0:
        print(f"done with adjacency functons..")

    prefix_lists = []

    # OMLtop, bot
    for _prefix in ["OMLtop", "OMLbot"]:
        prefix_lists += [
            f"{_prefix}{iOML}-{ispar}"
            for iOML in range(1, nOML + 1)
            for ispar in range(1, nspars + 2)
        ]

    # ribs
    prefix_lists += [
        f"rib{irib}-{ispar}"
        for irib in range(1, nribs + 1)
        for ispar in range(1, nspars + 2)
    ]
    # spars
    prefix_lists += [
        f"{spar_prefix}-{iOML}"
        for spar_prefix in spar_prefixes
        for iOML in range(1, nOML + 1)
    ]

    n2 = len(prefix_lists)
    for j, prefix in enumerate(prefix_lists):

        if comm.rank == 0:
            print(f"reg rel comp funcs {ncomp}")
            if j < n2 - 1:
                print("\033[1A", end="")

        skin_var = f2f_model.get_variables(f"{prefix}-pthick")
        sthick_var = f2f_model.get_variables(f"{prefix}-sthick")
        sheight_var = f2f_model.get_variables(f"{prefix}-sheight")
        spitch_var = f2f_model.get_variables(f"{prefix}-spitch")

        # stiffener - skin thickness adjacency here
        if skin_var is not None and sthick_var is not None:
            adj_value = thick_adj
            adj_constr = 15.0 * skin_var - sthick_var
            adj_constr.set_name(f"{prefix}-skin_stiff_T").optimize(
                lower=0.0, scale=10.0, objective=False
            ).setup_sparse_gradient(f2f_model)

            opt_manager.register_sparse_constraint(opt_problem, adj_constr)

            del adj_constr
            ncomp += 1

            # minimum stiffener spacing pitch > sheight
        if spitch_var is not None and sheight_var is not None:
            min_spacing_constr = spitch_var - sheight_var
            min_spacing_constr.set_name(f"{prefix}-sspacing").optimize(
                lower=0.0, scale=1.0, objective=False
            ).setup_sparse_gradient(f2f_model)

            opt_manager.register_sparse_constraint(opt_problem, min_spacing_constr)

            del min_spacing_constr
            ncomp += 1

        # minimum stiffener AR
        if sheight_var is not None and sthick_var is not None:
            min_stiff_AR = sheight_var - 5.0 * sthick_var
            min_stiff_AR.set_name(f"{prefix}-minstiffAR").optimize(
                lower=0.0, scale=1.0, objective=False
            ).setup_sparse_gradient(f2f_model)

            opt_manager.register_sparse_constraint(opt_problem, min_stiff_AR)

            del min_stiff_AR
            ncomp += 1

        # maximum stiffener AR (for regions with tensile strains where crippling constraint won't be active)
        if sheight_var is not None and sthick_var is not None:
            max_stiff_AR = sheight_var - 30.0 * sthick_var
            max_stiff_AR.set_name(f"{prefix}-maxstiffAR").optimize(
                upper=0.0, scale=1.0, objective=False
            ).setup_sparse_gradient(f2f_model)

            opt_manager.register_sparse_constraint(opt_problem, max_stiff_AR)

            del max_stiff_AR
            ncomp += 1

    if comm.rank == 0:
        print(f"number of composite functions = {ncomp}", flush=True)
