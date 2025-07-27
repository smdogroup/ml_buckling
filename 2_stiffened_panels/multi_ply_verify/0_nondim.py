import ml_buckling as mlb
from mpi4py import MPI
import argparse
import numpy as np
from _src import compute_stiffened_ABD

"""
goal here is to figure out the best pair of single vs multi-ply to have same nondim params
"""

comm = MPI.COMM_WORLD

# inputs
# ------------------------

# tune these values in order to find good match

# theta, AR_mult, hw_mult = 0.0, 2.4, 1.3 # are good values to match the multi ply [0,90] case.. although zeta is slightly diff..
# theta, AR_mult, hw_mult = 

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument(
    "--theta", default=0.0, type=float
)
parent_parser.add_argument(
    "--AR_mult", default=2.4, type=float
)
parent_parser.add_argument(
    "--hw_mult", default=1.3, type=float
)
args = parent_parser.parse_args()

# -------------------------

for mcase in [1,2]:

    # _h_w, _AR = 0.02, 1.0 # lightly stiffened case
    _h_w, _AR = 0.04, 1.0 # more largely stiffened

    if mcase == 1: # single ply
        ply_angles_deg = np.array([args.theta])
        ply_fractions = np.array([1.0])

        h_w, AR = _h_w * args.hw_mult, _AR * args.AR_mult 

    else:
        # ply_angles = np.deg2rad([0.0, 45.0, -45.0, 90.0])
        # ply_fractions = np.array([0.25, 0.25, 0.25, 0.25])
        ply_angles_deg = np.array([0.0, 90.0, 90.0, 0.0]) # sym laminate
        ply_fractions = np.array([0.25]*4)

        h_w, AR = _h_w, _AR


    # constants / dependent values
    stiff_AR, plate_SR, b = 20.0, 100.0, 1.0
    h = b / plate_SR
    t_w = h_w / stiff_AR
    a = b * AR
    sp = b / 2.0 # for 1 stiffener

    # now compute the nondim params
    # -----------------------------

    # first the ABD matrix where B = 0
    ABD_plate, z_c, E1s, Is = compute_stiffened_ABD(
        E1=117.9e9, 
        E2=9.7e9, 
        G12=4.8e9, 
        nu12=0.35,
        plate_thickness=h,
        ply_angles_deg=ply_angles_deg,
        ply_fractions=ply_fractions,
        h_w=h_w, t_w=t_w,
        nstiff=1,
    )

    # print(f"{ABD_plate=}")

    # now compute the non-dim params now
    A = ABD_plate[:3, :3]
    D = ABD_plate[3:,3:]
    rho0 = a / b * (D[1,1] / D[0,0])**0.25
    xi = (D[0,1] + 2.0 * D[-1,-1]) / np.sqrt(D[0,0] * D[1,1])
    gamma = E1s * Is / sp / D[0,0]
    zeta = (h / b)**2 * A[0,0] / A[-1,-1]

    A_arr = [A[0,0], A[0,1], A[1,1], A[-1,-1]]
    D_arr = [D[0,0], D[0,1], D[1,1], D[-1,-1]]

    if mcase == 1:
        print("single ply:")
    else:
        print("multi ply:")

    print(f"\t{rho0=:.3e}, {xi=:.3e}")
    print(f"\t{gamma=:.3e}, {zeta=:.3e}")

    print(f"\tangles: ", end="")
    for i in range(ply_angles_deg.shape[0]):
        print(f"{np.rad2deg(ply_angles_deg[i]):.1f}, ", end="")
    print("")

    print(f"\tA_arr: ", end="")
    for i in range(4):
        print(f"{A_arr[i]:.3e}, ", end="")
    print(f"\n\tD_arr: ", end="")
    for i in range(4):
        print(f"{D_arr[i]:.3e}, ", end="")
    print("\n------------------------------\n")