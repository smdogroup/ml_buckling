import os
from funtofem import *

def optimize_model(
        comm,
        driver, 
        args,
        file_prefix, 
        verify_level:int=0,
        debug=False,
    ):

    from pyoptsparse import SNOPT, Optimization

    if comm.rank == 0 and not os.path.exists("design/opt"):
        os.mkdir("design/opt")

    design_out_file = f"design/{file_prefix}_design.txt"
    history_file = f"design/opt/{file_prefix}.hst"

    # if args.reload_design:
    #     f2f_model.read_design_variables_file(comm, design_in_file)

    # if args.coldstart:
    #     f2f_model.read_design_variables_file(comm, design_out_file)

    manager = OptimizationManager(
        driver,
        design_out_file=design_out_file,
        hot_start=args.hotstart,
        debug=debug,
        hot_start_file=history_file,
        sparse=True,
        plot_hist=args.plot_hist
    )

    # create the pyoptsparse optimization problem
    opt_problem = Optimization("aob-sizing", manager.eval_functions)

    # add funtofem model variables to pyoptsparse
    manager.register_to_problem(opt_problem)

    # run an SNOPT optimization
    snoptimizer = SNOPT(
        options={
            "Print frequency": 1000,
            "Summary frequency": 10000000,
            "Major feasibility tolerance": 1e-6,
            "Major optimality tolerance": 1e-4,
            "Verify level": verify_level,
            "Major iterations limit": 15000,
            "Minor iterations limit": 150000000,
            "Iterations limit": 100000000,
            # "Major step limit": 5e-2, # had this off I think (but this maybe could be on)
            "Nonderivative linesearch": True,  # turns off derivative linesearch
            "Linesearch tolerance": 0.9,
            "Difference interval": 1e-6,
            "Function precision": 1e-10,
            "New superbasics limit": 2000,
            "Penalty parameter": 1.0,  # had this off for faster opt in the single panel case
            # however ksfailure becomes too large with this off. W/ on merit function goes down too slowly though
            # try intermediate value btw 0 and 1 (smaller penalty)
            # this may be the most important switch to change for opt performance w/ ksfailure in the opt
            # TODO : could try higher penalty parameter like 50 or higher and see if that helps reduce iteration count..
            #   because it often increases the penalty parameter a lot near the optimal solution anyways
            "Scale option": 1,
            "Hessian updates": 40,
            'Proximal iterations limit' : 0, # turn this off so it doesn't clip linear bounds
            "Print file": os.path.join(f"design/opt/{file_prefix}_SNOPT_print.out"),
            "Summary file": os.path.join(f"design/{file_prefix}_SNOPT_summary.out"),
        }
    )

    sol = snoptimizer(
        opt_problem,
        sens=manager.eval_gradients,
        storeHistory=history_file,  # None
        hotStart=history_file if args.hotstart else None,
    )

    return sol
