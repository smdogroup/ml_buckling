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
parent_parser.add_argument("--clear", type=bool, default=False)

args = parent_parser.parse_args()

args.load = "Nxy"

inner_ct = 0

for material in mlb.CompositeMaterial.get_materials()[1:]:
    for ply_angle in np.linspace(0.0, 90.0, 8)[3:]:
        plate_material = material(
            ply_angles=[ply_angle], ply_fractions=[1.0], ref_axis=[1, 0, 0]
        )
        stiff_material = plate_material

        log_SR_vec = np.linspace(np.log(10.0), np.log(200.0), 5)
        SR_vec = np.exp(log_SR_vec)
        for SR in SR_vec[::-1]:

            # choose plate height is 0.01
            h = 0.005
            b = h * SR

            # stiffener heights and spacing
            log_SHR = np.linspace(np.log(1.0), np.log(8.0), 8)
            SHR_vec = np.exp(log_SHR)
            SHR_vec = SHR_vec[::-1]

            for SHR in SHR_vec:

                SHR = 0.01

                for num_stiff in [1, 3, 5]:

                    log_AR_vec = np.linspace(np.log(0.2), np.log(8.0), 15)
                    AR_vec = np.exp(log_AR_vec)

                    for AR in AR_vec[3:4]:

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
                        _nelems = 6000
                        MIN_Y = 20 / geometry.num_local
                        MIN_Z = 10 #5
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

                        # do linear static analysis
                        stresses = stiffened_plate.run_static_analysis(write_soln=True)
                
                        tacs_eigvals, errors = stiffened_plate.run_buckling_analysis(
                            sigma=5.0, num_eig=50, write_soln=True
                        )
                        stiffened_plate.post_analysis()

                        if comm.rank == 0:
                            stiffened_plate.print_mode_classification()

                        # if abs(errors[0]) > 1e-7: continue

                        global_lambda_star = stiffened_plate.min_global_mode_eigenvalue
                        if comm.rank == 0:
                            print(f"stresses = {stresses}")
                            print(f"global lambda = {global_lambda_star}")

                        # exit after first model
                        exit()