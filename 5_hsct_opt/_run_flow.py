"""
Run a FUN3D analysis with the Fun3dOnewayDriver
"""

from funtofem import *
from mpi4py import MPI
import os

#case = "turbulent"
case = "inviscid"

comm = MPI.COMM_WORLD

f2f_model = FUNtoFEMmodel("hsct_flow")
wing = Body.aerothermoelastic("wing", boundary=3)
wing.register_to(f2f_model)

# make a funtofem scenario
if case == "inviscid":
    climb = Scenario.steady("climb_inviscid", steps=3000)  # 2000
elif case == "turbulent":
    climb = Scenario.steady("climb_turb", steps=1000)
Function.lift().register_to(climb)
Function.drag().register_to(climb)
climb.set_temperature(T_ref=216, T_inf=216)
climb.set_flow_ref_vals(qinf=3.16e4)
climb.register_to(f2f_model)


# DISCIPLINE INTERFACES AND DRIVERS
# -----------------------------------------------------

solvers = SolverManager(comm)
solvers.flow = Fun3d14Interface(
    comm, f2f_model, fun3d_dir="cfd"
)
my_transfer_settings = TransferSettings(npts=200)
fun3d_driver = OnewayAeroDriver(
    solvers, f2f_model, transfer_settings=my_transfer_settings
)
fun3d_driver.solve_forward()

# write an aero loads file
if case == "inviscid":
    aero_loads_file = os.path.join(os.getcwd(), "cfd", "loads", "uncoupled_loads.txt")
elif case == "turbulent":
    aero_loads_file = os.path.join(
        os.getcwd(), "cfd", "loads", "uncoupled_turb_loads.txt"
    )
f2f_model.write_aero_loads(comm, aero_loads_file)
