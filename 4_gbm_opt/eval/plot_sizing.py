from funtofem import *

# 1 : plot 1_sizing_opt_local.py results
# ---------------------------------------------------------------------------
plotter = PlotManager.from_hist_file(
    "ML-oneway_design.txt",
    accepted_names=["climb_turb-mass", "climb_turb-ksfailure"],
    plot_names=["mass", "ks-stress"],
    ignore_other_names=True,
)

# MAKE EACH PLOT FOR A DIFFERENT MODE
functions = []
Function.plot("ks-stress").optimize(scale=1.0, plot_name="ks-stress").register_to(
    plotter
)
plotter.add_constraint(value=1.0, name="ks-constr")
Function.plot("mass").optimize(scale=1e-4, plot_name="wing-weight [10 kN]").register_to(
    plotter
)

plotter(
    plot_name="sizing-hist.png",
    legend_frac=0.9,
    yaxis_name="function values",
    color_offset=0,
    niceplots_style="doumont-light",
    yscale_log=True,
)
