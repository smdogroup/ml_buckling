"""
Sean Engelstad
April 2024, GT SMDO Lab
Goal is to generate a dataset for pure uniaxial and pure shear compression for stiffener crippling data.
Simply supported only and stiffener crippling modes are rejected.

NOTE : copy u*iHat+v*jHat+w*kHat for paraview
"""

import ml_buckling as mlb
import pandas as pd
import numpy as np, os, argparse
from mpi4py import MPI

comm = MPI.COMM_WORLD

# argparse
parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--load", type=str)
parent_parser.add_argument("--clear", type=bool, default=False)

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

if args.clear and comm.rank == 0:
    if os.path.exists(train_csv_path):
        os.remove(train_csv_path)
    if os.path.exists(raw_csv_path):
        os.remove(raw_csv_path)
comm.Barrier()

inner_ct = 0

for material in mlb.CompositeMaterial.get_materials():
    for ply_angle in np.linspace(0.0, 90.0, 8):
        plate_material = material(
            ply_angles=[ply_angle], ply_fractions=[1.0], ref_axis=[1, 0, 0]
        )
        stiff_material = plate_material

        log_SR_vec = np.linspace(np.log(10.0), np.log(200.0), 5)
        SR_vec = np.exp(log_SR_vec)
        for SR in SR_vec[::-1]:

            # choose plate height is 0.01
            h = 1.0
            b = h * SR

            # stiffener heights and spacing
            log_SHR = np.linspace(np.log(0.1), np.log(8.0), 8)
            SHR_vec = np.exp(log_SHR)
            SHR_vec = SHR_vec[::-1]

            for SHR in SHR_vec:

                for num_stiff in [1, 3, 5]:

                    log_AR_vec = np.linspace(np.log(0.2), np.log(5.0), 15)
                    AR_vec = np.exp(log_AR_vec)

                    for AR in AR_vec:

                        # temporarily set AR to reasonable value
                        #AR = 3.0
                        a = AR * b

                        # use stiffener height ratio to determine the stiffener height
                        h_w = h * SHR
                        stiff_AR = 3.0 #1.0
                        t_w = h_w / stiff_AR

                        geometry = mlb.StiffenedPlateGeometry(
                            a=a,
                            b=b,
                            h=h,
                            num_stiff=num_stiff,
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

                        # choose a number of elements in each direction
                        _nelems = 8000
                        MIN_Y = 30 / geometry.num_local
                        MIN_Z = 15 #5
                        N = geometry.num_local
                        AR_s = geometry.a / geometry.h_w
                        #print(f"AR = {AR}, AR_s = {AR_s}")
                        nx = np.ceil(np.sqrt(_nelems / (1.0/AR + (N-1) / AR_s)))
                        ny = max(np.ceil(nx / AR / N), MIN_Y)
                        nz = max(np.ceil(nx / AR_s), MIN_Z)
                        print(f"Stage 1 : nx {nx}, ny {ny}, nz {nz}")

                        # if nz < MIN_Z: # need at least this many elements through the stiffener for good aspect ratio
                        #     nz = MIN_Z
                        #     nx = np.ceil(AR_s * nz)
                        #     ny = np.ceil(nx / AR / N)
                        
                        # print(f"Stage 2 : nx {nx}, ny {ny}, nz {nz}")

                        check_nelems = N * nx * ny + (N-1) * nx * nz
                        #print(f"check nelems = {check_nelems}")

                        print(f"check nelems = {check_nelems}")

                        stiffened_plate.pre_analysis(
                            exx=stiffened_plate.affine_exx
                            if args.load == "Nx"
                            else 0.0,
                            exy=stiffened_plate.affine_exy
                            if args.load == "Nxy"
                            else 0.0,
                            clamped=False,
                            nx_plate=int(nx),
                            ny_plate=int(ny),
                            nz_stiff=int(nz),
                            # global_mesh_size=global_mesh_size,
                            # edge_pt_min=5,
                            # edge_pt_max=50,
                            _make_rbe=True,  # True
                        )

                        if comm.rank == 0:
                            print(stiffened_plate)
                        # exit()

                        lam_min, mode_type = stiffened_plate.predict_crit_load(
                            exx=stiffened_plate.affine_exx
                            if args.load == "Nx"
                            else 0.0,
                            exy=stiffened_plate.affine_exy
                            if args.load == "Nxy"
                            else 0.0,
                            output_global=True,
                        )
                        lam_min2 = stiffened_plate.predict_crit_load_old(
                            exx=stiffened_plate.affine_exx
                            if args.load == "Nx"
                            else 0.0,
                            exy=stiffened_plate.affine_exy
                            if args.load == "Nxy"
                            else 0.0,
                            output_global=True,
                        )

                        tacs_eigvals, errors = stiffened_plate.run_buckling_analysis(
                            sigma=5.0, num_eig=50, write_soln=True
                        )
                        stiffened_plate.post_analysis()

                        if comm.rank == 0:
                            stiffened_plate.print_mode_classification()

                        # if abs(errors[0]) > 1e-7: continue

                        global_lambda_star = stiffened_plate.min_global_mode_eigenvalue
                        if global_lambda_star is None:
                            continue  # no global modes appeared

                        if not (0.5 <= global_lambda_star <= 200.0):
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
                            "pred_type" : [mode_type],
                            "pred_lam_old" : [lam_min2],
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
                        data_dict["delta"] = [stiffened_plate.delta]
                        data_dict["n_stiff"] = [num_stiff]
                        data_dict["elem_list"] = [[int(nx),int(ny),int(nz)]]
                        data_dict["nelem"] = [int(check_nelems)]

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
