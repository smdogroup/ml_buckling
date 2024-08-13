from funtofem import *
import matplotlib.pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD

f2f_model = FUNtoFEMmodel("hsct-wing")
wing = Body.aeroelastic("wing")
wing.register_to(f2f_model)
climb = Scenario.steady("climb_turb", steps=1)
climb.register_to(f2f_model)
# wing.initialize_variables(climb)

f2f_model._read_aero_loads(comm, "uncoupled_turb_loads.txt")

# view the aero mesh
aero_X = wing.aero_X
ax = plt.figure().add_subplot(projection="3d")
ax.scatter(aero_X[0::3], aero_X[1::3], aero_X[2::3])
ax.set_aspect("equal")
plt.show()
