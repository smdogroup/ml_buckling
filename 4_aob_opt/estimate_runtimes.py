import numpy as np

orig_procs = 4
new_procs = 48
proc_adjust = orig_procs / new_procs

# AOB runtimes
# ------------

num_panels = 111
orig_lstatic_elems = 70e3
new_lstatic_elems = 4e3 * num_panels
orig_lstatic_cost = 28.5 # hrs
AOB_new_lstatic_cost = orig_lstatic_cost * (new_lstatic_elems / orig_lstatic_elems)**1.5 * proc_adjust
print(f"{AOB_new_lstatic_cost=}")

# online panel buckling
num_func_evals = 2300
hun_eigval_buckle_cost = 11.589 / 3600 # sec to hrs for 100 eigenvalues, both axial and shear at 8k elements
orig_buckle_elems = 4e3
oneeigval_buckle_cost = hun_eigval_buckle_cost / 10 # scale is like sqrt(num eigvals)
AOB_online_panel_cost = hun_eigval_buckle_cost * num_panels * num_func_evals * proc_adjust
print(f"{AOB_online_panel_cost=}")

# full wingbox buckling
new_elems = num_panels * 4e3
AOB_single_buckle_cost = oneeigval_buckle_cost * (new_elems / orig_buckle_elems)**1.5
AOB_full_buckle_cost = AOB_single_buckle_cost * num_func_evals
AOB_full_wingbox_cost = AOB_new_lstatic_cost + AOB_full_buckle_cost
AOB_full_wingbox_cost *= proc_adjust
print(f'{AOB_full_buckle_cost=}')
print(f"{AOB_full_wingbox_cost=}")


# HSCT case
# ---------

num_panels = 1614
orig_lstatic_elems = 20e3
new_lstatic_elems = 4e3 * num_panels
orig_lstatic_cost = 43 # hrs
HSCT_new_lstatic_cost = orig_lstatic_cost * (new_lstatic_elems / orig_lstatic_elems)**1.5
print(f"{HSCT_new_lstatic_cost=:.4e}")

# online panel buckling
num_func_evals = 10100
HSCT_online_panel_cost = hun_eigval_buckle_cost * num_panels * num_func_evals * proc_adjust
print(f"{HSCT_online_panel_cost=:.4e}")

# full wingbox buckling
new_elems = num_panels * 4e3
HSCT_single_buckle_cost = oneeigval_buckle_cost * (new_elems / orig_buckle_elems)**1.5
HSCT_full_buckle_cost = HSCT_single_buckle_cost * num_func_evals
HSCT_full_wingbox_cost = HSCT_new_lstatic_cost + HSCT_full_buckle_cost
HSCT_full_wingbox_cost *= proc_adjust
print(f'{HSCT_full_buckle_cost=:.4e}')
print(f"{HSCT_full_wingbox_cost=:.4e}")

# total costs
panel_buckle = AOB_online_panel_cost + HSCT_online_panel_cost
print(f"{panel_buckle=:.4e}")
full_wingbox = AOB_full_wingbox_cost + HSCT_full_wingbox_cost
print(f"{full_wingbox=:.4e}")