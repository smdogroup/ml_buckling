from tacs import pyTACS
from mpi4py import MPI
import os, numpy as np

comm = MPI.COMM_WORLD

# running TACS analysis manually since TACS .f5 output from FUNtoFEM 
# TACSinterface is being buggy right now

# read the sizing txt file
hdl = open("../sizing.txt", "r")
lines = hdl.readlines()
hdl.close()
des_dict = {}
for line in lines:
    if "var" in line:
        chunks = line.split(" ")
        name = chunks[1]
        value = float(chunks[2])
        des_dict[name] = value

#print(f"names3 = {names3}")
xarr = np.array([0.002])
#print(f"xarr = {xarr}")
#exit()

fea_assembler = pyTACS("tacs.dat", comm=comm)
fea_assembler.initialize()
SPs = fea_assembler.createTACSProbsFromBDF()

for caseID in SPs:
    SP = SPs[caseID]

    # modify the xarr of the struct problem
    if comm.rank == 0:
        SP.setDesignVars(xarr)
    comm.Barrier()

    #SP.solve()
    SP.writeSolution(baseName="htail")
