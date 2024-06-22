from tacs import pyTACS
from mpi4py import MPI
import os, numpy as np

comm = MPI.COMM_WORLD

# running TACS analysis manually since TACS .f5 output from FUNtoFEM 
# TACSinterface is being buggy right now

# read the sizing txt file
# hdl = open("../sizing.txt", "r")
# lines = hdl.readlines()
# hdl.close()
# des_dict = {}
# for line in lines:
#     if "var" in line:
#         chunks = line.split(" ")
#         name = chunks[1]
#         value = float(chunks[2])
#         des_dict[name] = value

# names = list(des_dict.keys())
# names2 = []
# for name in names:
#     if "OML" in name or "LE" in name and not("spar" in name):
#         names2 += [name]
# names3 = np.sort(np.array(names2))
# #print(f"names3 = {names3}")
# xarr = np.array([des_dict[name] for name in names3])

fea_assembler = pyTACS("tacs.dat", comm=comm)
fea_assembler.initialize()
SPs = fea_assembler.createTACSProbsFromBDF()

for caseID in SPs:
    SP = SPs[caseID]

    # modify the xarr of the struct problem
    # if comm.rank == 0:
    #     SP.setDesignVars(xarr)
    # comm.Barrier()

    #SP.solve()
    SP.writeSolution(baseName="upper-OML")