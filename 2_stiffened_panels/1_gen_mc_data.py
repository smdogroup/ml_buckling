"""
Sean Engelstad
April 2024, GT SMDO Lab
Goal is to generate a dataset for pure uniaxial and pure shear compression for stiffener crippling data.
Simply supported only and stiffener crippling modes are rejected.
"""

import ml_buckling as mlb
import pandas as pd
import numpy as np, os, argparse
from mpi4py import MPI

comm = MPI.COMM_WORLD

# argparse
parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--load", type=str)

args = parent_parser.parse_args()

assert args.load in ["Nx", "Nxy"]

train_csv = args.load + "_stiffened.csv"
raw_csv = args.load + "_raw_stiffened.csv"
cpath = os.path.dirname(__file__)
data_folder = os.path.join(cpath, "data")
if not os.path.exists(data_folder) and comm.rank == 0:
    os.mkdir(data_folder)

train_csv_path = os.path.join(data_folder, train_csv)
raw_csv_path = os.path.join(data_folder, raw_csv)

inner_ct = 0

for material in mlb.CompositeMaterial.get_materials():
    for ply_angle in np.linspace(0.0, 90.0, 8):
        plate_material = material(
            ply_angles=[ply_angle], ply_fractions=[1.0], ref_axis=[1, 0, 0]
        )
        stiff_material = plate_material

        log_SR_vec = np.linspace(np.log(10.0), np.log(200.0), 5)
        SR_vec = np.exp(log_SR_vec)
        for SR in SR_vec:

            # choose plate height is 0.01
            h = 0.01
            b = h * SR

            # stiffener heights and spacing
            log_SHR = np.linspace(np.log(0.001), np.log(0.4), 5)
            SHR_vec = np.exp(log_SHR)
            # SHR_vec = SHR_vec[::-1]

            for SHR in SHR_vec:

                # use stiffener height ratio to determine the stiffener height
                h_w = b * SHR
                stiff_AR = 1.0
                t_w = h_w / stiff_AR

                for num_stiff in [1, 3, 5]:

                    log_AR_vec = np.linspace(np.log(0.2), np.log(5.0), 15)
                    AR_vec = np.exp(log_AR_vec)

                    for AR in AR_vec:

                        # temporarily set AR to reasonable value
                        AR = 3.0

                        geometry = mlb.StiffenedPlateGeometry(
                            a=AR * SR * h,
                            b=SR * h,
                            h=h,
                            num_stiff=num_stiff,
                            w_b=1.0,
                            t_b=0.0,
                            h_w=h_w,
                            t_w=t_w,
                        )

                        stiffened_plate = mlb.StiffenedPlateAnalysis(
                            comm=comm,
                            geometry=geometry,
                            stiffener_material=stiff_material,
                            plate_material=plate_material,
                            name="mc",
                        )

                        valid_affine_AR = (
                            0.05 <= stiffened_plate.affine_aspect_ratio <= 20.0
                        )
                        if not valid_affine_AR:
                            continue

                        # choose a number of elements
                        # nelem = 2000
                        # global_mesh_size = np.sqrt(geometry.a * (geometry.b + geometry.num_stiff * geometry.h_w) / nelem) * 4
                        global_mesh_size = 0.01
                        print(f"global mesh size = {global_mesh_size}")

                        stiffened_plate.pre_analysis(
                            global_mesh_size=global_mesh_size,
                            exx=stiffened_plate.affine_exx
                            if args.load == "Nx"
                            else 0.0,
                            exy=stiffened_plate.affine_exy
                            if args.load == "Nxy"
                            else 0.0,
                            clamped=False,
                            edge_pt_min=5,
                            edge_pt_max=50,
                            _make_rbe=False,  # True
                        )

                        print(stiffened_plate)
                        # exit()

                        lam_min, mode_type = stiffened_plate.predict_crit_load(
                            exx=stiffened_plate.affine_exx
                            if args.load == "Nx"
                            else 0.0,
                            exy=stiffened_plate.affine_exy
                            if args.load == "Nxy"
                            else 0.0,
                        )

                        print(f"lam min predicted = {lam_min} of type {mode_type}")

                        tacs_eigvals, errors = stiffened_plate.run_buckling_analysis(
                            sigma=10.0, num_eig=50, write_soln=True
                        )
                        stiffened_plate.post_analysis()

                        if comm.rank == 0:
                            stiffened_plate.print_mode_classification()

                        # if abs(errors[0]) > 1e-7: continue

                        global_lambda_star = stiffened_plate.min_global_mode_eigenvalue
                        if global_lambda_star is None:
                            continue  # no global modes appeared

                        if not (0.5 <= global_lambda_star <= 50.0):
                            print(
                                f"Warning global mode eigenvalue {global_lambda_star} not in [0.5, 50.0]"
                            )
                            continue

                        # save data to csv file otherwise because this data point is good
                        # record the model parameters
                        data_dict = {
                            # training parameter section
                            "rho_0": [stiffened_plate.affine_aspect_ratio],
                            "xi": [stiffened_plate.xi_plate],
                            "gamma": [stiffened_plate.gamma],
                            "zeta": [stiffened_plate.zeta_plate],
                            "lambda_star": [np.real(global_lambda_star)],
                            "pred_lam": [lam_min],
                        }

                        # write to the training csv file
                        inner_ct += 1  # started from 0, so first time is 1
                        if comm.rank == 0:
                            df = pd.DataFrame(data_dict)
                            if inner_ct == 1 and not (os.path.exists(train_csv_path)):
                                df.to_csv(
                                    train_csv_path, mode="w", index=False, header=True
                                )
                            else:
                                df.to_csv(
                                    train_csv_path, mode="a", index=False, header=False
                                )

                        comm.Barrier()

                        # add raw data section
                        data_dict["material"] = [plate_material.material_name]
                        data_dict["ply_angle"] = [ply_angle]
                        data_dict["SR"] = [SR]
                        data_dict["AR"] = [AR]
                        data_dict["h"] = [h]
                        data_dict["SHR"] = [SHR]
                        data_dict["SAR"] = [stiff_AR]
                        data_dict["n_stiff"] = [num_stiff]

                        # write to the csv file for raw data
                        if comm.rank == 0:
                            df = pd.DataFrame(data_dict)
                            if inner_ct == 1 and not (os.path.exists(raw_csv_path)):
                                df.to_csv(
                                    raw_csv_path, mode="w", index=False, header=True
                                )
                            else:
                                df.to_csv(
                                    raw_csv_path, mode="a", index=False, header=False
                                )

                        # MPI COMM Barrier in case running with multiple procs
                        comm.Barrier()

                        # if inner_ct == 1:
                        #    exit() # temporary debug exit()
