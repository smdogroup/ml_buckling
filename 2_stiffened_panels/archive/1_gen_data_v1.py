import numpy as np
import ml_buckling as mlb
import pandas as pd
from mpi4py import MPI
import os, argparse, sys

# local src for this case
sys.path.append("src/")
from buckling_analysis import get_buckling_load 

comm = MPI.COMM_WORLD
np.random.seed(123456)

"""
main dataset generation for finite element dataset of paper
includes unstiffened, 1 stiffener and 9 stiffener models
"""

# argparse
# --------

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument(
    "--clear", default=False, action=argparse.BooleanOptionalAction
)
parent_parser.add_argument(
    "--axial", default=True, action=argparse.BooleanOptionalAction
)
parent_parser.add_argument(
    "--metal", default=True, action=argparse.BooleanOptionalAction
)
parent_parser.add_argument(
    "--debug", default=False, action=argparse.BooleanOptionalAction
)
args = parent_parser.parse_args()

# setup csv filepath
# ------------------

prefix = "Nx" if args.axial else "Nxy"
train_csv = f"{prefix}_stiffened.csv"
raw_csv = f"{prefix}_raw_stiffened.csv"

cpath = os.path.dirname(__file__)
data_folder = os.path.join(cpath, "data")
if not os.path.exists(data_folder) and comm.rank == 0:
    os.mkdir(data_folder)

train_csv_path = os.path.join(data_folder, train_csv)
raw_csv_path = os.path.join(data_folder, raw_csv)

if args.clear and comm.rank == 0:
    if os.path.exists(train_csv_path):
        os.remove(train_csv_path)
    if os.path.exists(raw_csv_path):
        os.remove(raw_csv_path)
comm.Barrier()

# main
if __name__ == "__main__":
    # only generates stiffened data, unstiffened we can downsample but we already have that data
    
    logrho_vec = np.linspace(-2.0, 2.0, 20) #ln(rho0 or rho0^*)
    loggamma_vec = np.linspace(0.1, 3.0, 10) # ln(1+gamma)
    # loggamma_vec = np.linspace(0.01, 3.0, 10) # ln(1+gamma)

    composite_materials = mlb.CompositeMaterial.get_materials()

    num_models = 0
    while (num_models < 1000): # 1000 each for metal + then composites

        # loop over rho0, gamma
        # ---------------------

        failed_in_a_row = 0

        for log_gamma in loggamma_vec:
            gamma = np.exp(log_gamma) - 1.0 # since log_gamma = np.log(1+gamma)

            prev_eig_dict = None

            # if failed_in_a_row > 15 and not(args.metal):
            #     # composites can start to fail a bunch with stiffener crippling
            #     break # restart gamma loop

            # choosing randomness here, so that consistent xi, zeta in the inner rho0 loops
            # otherwise mode tracking for very low rho0 doesn't work
            # -----------------------------------------------------

            ply_angle = None; plate_slenderness = None; plate_material = None
            if comm.rank == 0:

                # choose random ply angle 0 to 90
                ply_angle = np.random.uniform(0.0, 90.0) # in deg
                
                # choose random slenderness 10 to 200
                log_slenderness = np.random.uniform(np.log(10.0), np.log(200.0))
                plate_slenderness = np.exp(log_slenderness)

                # choose a random material
                if args.metal:
                    E = 138e9; nu = 0.3
                    G = E / 2.0 / (1 + nu)
                    plate_material = mlb.CompositeMaterial(
                        E11=E,  # Pa
                        E22=E,
                        G12=G,
                        nu12=nu,
                        ply_angles=[0], # ply angle doesn't really matter with metal
                        ply_fractions=[1.0],
                        ref_axis=[1, 0, 0],
                    )
                else:
                    composite_material = np.random.choice(np.array(composite_materials))

                    plate_material = composite_material(
                        ply_angles=[ply_angle],
                        ply_fractions=[1.0],
                        ref_axis=[1.0, 0.0, 0.0],
                    )

            ply_angle = comm.bcast(ply_angle, root=0)
            plate_slenderness = comm.bcast(plate_slenderness, root=0)
            plate_material = comm.bcast(plate_material, root=0)

            for log_rho in logrho_vec[::-1]: # go in reverse order so that mode tracking can be done
                if args.axial: # log_rho = ln(rho0^*) = ln(rho0 / sqrt[4]{1+gamma})
                    rho0_star = np.exp(log_rho)
                    rho0 = rho0_star * (1.0 + gamma)**0.25
                else:
                    rho0 = np.exp(log_rho)

                # choose the number of stiffeners based on (log_rho, log_gamma)
                # ------------------------------------------------------------

                # approx boundary of 1 stiffener so that (-2,0) and (0.5, 1.5) from stiffener study are on boundary

                # debug
                if args.debug:
                    # gamma = 3.0; rho0 = 1.0
                    gamma = 0.105; rho0 = 0.26
                    log_gamma = np.log(1+gamma); log_rho = np.log(rho0)
                    if args.axial: log_rho -= 0.25 * log_gamma

                use_single_stiffener = log_gamma < 0.6 * (log_rho + 1.5)
                # use_single_stiffener = log_rho < 0.0
                num_stiff = 1 if use_single_stiffener else 9

                if comm.rank == 0:
                    material_name = "metal" if args.metal else composite_material.__name__
                    ply_angle = 0.0 if args.metal else ply_angle
                    print(f"{gamma=} {rho0=} {num_stiff=} {ply_angle=} {material_name=} {plate_slenderness=}")

                # run the buckling analysis
                # -------------------------

                eig_CF, eig_FEA, stiff_analysis, new_eig_dict, dt = \
                get_buckling_load(
                    comm,
                    rho0=rho0,
                    gamma=gamma,
                    plate_slenderness=plate_slenderness,
                    num_stiff=num_stiff,
                    plate_material=plate_material,
                    nelems=2000, #2000
                    prev_eig_dict=None, #prev_eig_dict,
                    is_axial=args.axial,
                    solve_buckling=True,
                    debug=args.debug,
                    b=1.0,
                    stiff_AR=15.0 # works better than 20.0 for getting more high gamma, low rho0 data
                )

                if comm.rank == 0:
                    print(f"{eig_CF=}, {eig_FEA=} {stiff_analysis.gamma=}")

                if eig_FEA is None:
                    eig_FEA = np.nan  # just leave value as almost zero..
                    failed_in_a_row += 1
                    continue
                else:
                    prev_eig_dict = new_eig_dict
                    failed_in_a_row = 0

                if args.debug: exit()

                # write out as you go so you can see the progress and if run gets killed you don't lose it all
                if comm.rank == 0:
                    my_material = "metal" if args.metal else composite_material.__name__

                    num_models += 1 # increment the number of models
                    raw_data_dict = {
                        # training parameter section
                        "rho_0": [stiff_analysis.affine_aspect_ratio],
                        "xi": [stiff_analysis.xi_plate],
                        "gamma": [stiff_analysis.gamma],
                        "zeta": [stiff_analysis.zeta_plate],
                        "num_stiff" : [num_stiff],
                        "material" : [my_material],
                        "eig_FEA": [np.real(eig_FEA)],
                        "eig_CF": [eig_CF],
                        "time" : [dt],
                    }
                    raw_df = pd.DataFrame(raw_data_dict)
                    first_write = num_models == 1 and args.clear
                    raw_df.to_csv(
                        raw_csv_path, mode="w" if first_write else "a", header=first_write,
                        float_format="%.6f"
                    )

                    # check mesh convergence..



