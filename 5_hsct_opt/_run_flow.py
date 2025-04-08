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
# climb = Scenario.steady("climb_inviscid", steps=3000)  # 2000
pullup = Scenario.steady(
    "pullup", steps=150, coupled=False, forward_coupling_frequency=30, uncoupled_steps=100
)
pushdown = Scenario.steady(
    "pushdown", steps=150, coupled=False, forward_coupling_frequency=30, uncoupled_steps=100
)

for my_scenario in [pullup, pushdown]:
    Function.lift().register_to(my_scenario)
    Function.drag().register_to(my_scenario)
    my_scenario.set_temperature(T_ref=300, T_inf=300)
    my_scenario.set_flow_ref_vals(qinf=2.2657e4)
    my_scenario.register_to(f2f_model)


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
    # aero_loads_file = os.path.join(os.getcwd(), "cfd", "loads", "uncoupled_loads.txt")
aero_loads_file = "cfd/loads/uncoupled_turb_loads.txt"
f2f_model.write_aero_loads(comm, aero_loads_file)
