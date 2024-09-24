from funtofem import *
import matplotlib.pyplot as plt, niceplots

import argparse

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument(
    "--useML", default=False, action=argparse.BooleanOptionalAction
)
args = parent_parser.parse_args()

# 1 : plot 1_sizing_opt_local.py results
# ---------------------------------------------------------------------------
m_case = "ML" if args.useML else "CF"
scenario_name = "climb-turb"
plotter = PlotManager.from_hist_file(
    "CF-oneway_design.txt" if m_case == "CF" else "ML-oneway_design.txt",
    accepted_names=["climb_turb-ksfailure", "climb_turb-mass"],
    plot_names=["ksfailure", "mass"],
    ignore_other_names=True,
)

# MAKE EACH PLOT FOR A DIFFERENT MODE
togw = Function.plot("mass").optimize(scale=1.0e-3 * 9.81).register_to(plotter)
ksfailure = Function.plot("ksfailure").optimize(scale=1.0).register_to(plotter)

# three color schemes from color scheme website https://coolors.co/palettes/popular/3%20colors
colors1 = ["#2b2d42", "#8d99ae", "#edf2f4"]
colors2 = ["#264653", "#2a9d8f", "#e9c46a"]
colors3 = ["#2e4057", "#048ba8", "#f18f01"]
colors4 = ["#c9cba3", "#ffe1a8", "#e26d5c"]
colors5 = ["#0b3954", "#bfd7ea", "#ff6663"]
colors6 = ["#0d3b66", "#faf0ca", "#f4d35e"]
colors7 = ["#000000", "#ff0000", "#ffe100"]
colors8 = ["#064789", "#427aa1", "#ebf2fa"]
colors9 = ["#0b132b", "#1c2541", "#3a506b"]
colors10 = ["#49beaa", "#456990", "#ef767a"]
colors11 = ["#1d2f6f", "#8390fa", "#fac748"]
six_colors = ["#264653", "#287271", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]

plt.figure("case1")
plt.style.use(niceplots.get_style())
fig, ax1 = plt.subplots(figsize=(8, 6))
# my_colors = niceplots.get_colors_list() #colors11
my_colors = colors3  # colors3, colors5
grey_colors = plt.cm.Greys(np.linspace(1.0, 0.5, 2))
plt.margins(x=0.05, y=0.05)
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Wing Weight (kN)", color=my_colors[0])
ax1.tick_params(axis="y", labelcolor=my_colors[0])
ax1.plot(
    plotter.iterations,
    plotter.get_hist_array(togw),
    "-",
    color=my_colors[0],
    label="Weight",
)

ax2 = ax1.twinx()

niter = len(plotter.iterations)
ax2.plot(
    plotter.iterations, np.ones((niter,)), color=grey_colors[0], linestyle="dashed"
)
ax2.plot(
    plotter.iterations, np.zeros((niter,)), color=grey_colors[1], linestyle="dashed"
)
ax2.plot(
    plotter.iterations,
    plotter.get_hist_array(ksfailure),
    "-",
    color=my_colors[1],
    label="KSfailure",
)
ax2.set_ylabel("KSFailure", color=my_colors[1])
ax2.tick_params(axis="y", labelcolor="k")
ax2.set_yscale("log")
ax2.set_ylim(1e-1, 1000)
plt.text(x=1000, y=0.7, s="ks-constr", color=grey_colors[0])

# legend_frac = 0.8
# box = ax2.get_position()
# ax2.set_position([box.x0, box.y0, box.width * legend_frac, box.height])
# ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5))

# plot the constraints
plt.margins(x=0.02, y=0.02)
# plt.legend()
plt.savefig("CF-opt-hist.png" if m_case == "CF" else "ML-opt-hist.png", dpi=400)
plt.close("case1")
