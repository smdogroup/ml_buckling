"""
Run a FUN3D analysis with the Fun3dOnewayDriver
"""

from funtofem import *
from mpi4py import MPI
import os

case = "turbulent"
# case = "inviscid"

comm = MPI.COMM_WORLD

f2f_model = FUNtoFEMmodel("hsct_flow")
wing = Body.aeroelastic("wing", boundary=3)
wing.register_to(f2f_model)

# make a funtofem scenario
cruise = Scenario.steady(
    "cruise_turb", steps=100, coupling_frequency=30, uncoupled_steps=100
)
cruise.set_stop_criterion(early_stopping=True)

mass = Function.mass().optimize(
    scale=1.0e-4, objective=True, plot=True, plot_name="mass"
)
ksfailure = Function.ksfailure(ks_weight=10.0, safety_factor=1.5).optimize(
    scale=1.0, upper=1.0, objective=False, plot=True, plot_name="ks-cruise"
)
cruise.include(ksfailure).include(mass)
cruise.set_temperature(T_ref=216, T_inf=216)
cruise.set_flow_ref_vals(qinf=3.1682e4)
cruise.register_to(f2f_model)

# DISCIPLINE INTERFACES AND DRIVERS
# -----------------------------------------------------

solvers = SolverManager(comm)
solvers.flow = Fun3d14Interface(
    comm,
    f2f_model,
    fun3d_dir="cfd",
    forward_stop_tolerance=1e-13,
    forward_min_tolerance=1e-8,
)
my_transfer_settings = TransferSettings(npts=200)
fun3d_driver = OnewayAeroDriver(
    solvers, f2f_model, transfer_settings=my_transfer_settings
)
fun3d_driver.solve_forward()

# write an aero loads file
aero_loads_file = os.path.join(
    os.getcwd(), "cfd", "loads", "cruise_turb_loads.txt"
)
f2f_model.write_aero_loads(comm, aero_loads_file)
