"""
Sean Engelstad, Feb 2024
GT SMDO Lab

For high slenderness, b/h in [50, 200]
and for ply angles 0 or 90 with xi around 0.3 to 0.8
the eigenvalues for stiffener crippling are above 1 sometimes and very large.
They don't have much solution error so it is kind of weird.
This doesn't make sense => see if needs mesh convergence study.
"""
import ml_buckling as mlb
import numpy as np
from mpi4py import MPI
import pandas as pd
import os

comm = MPI.COMM_WORLD

h = 1.0

# AR_vec = [11.7210]
# AR_vec = [0.2, 1, 2,4,6,8,10,20]
log_AR_vec = np.linspace(-1, 1.2, 15)
AR_vec = np.power(10.0, log_AR_vec)
inner_ct = 0

log_SR_vec = np.linspace(np.log(10.0), np.log(200.0), 6)
SR_vec = np.exp(log_SR_vec)
materials = mlb.UnstiffenedPlateAnalysis.get_materials()

for material in materials:

    for SR in SR_vec[::-1]:

        for ply_angle in np.linspace(0.0, 90.0, 10):

            rel_errs = []

            for AR in AR_vec:

                b = h * SR
                a = b * AR

                flat_plate = material(
                    comm=comm,
                    bdf_file="plate.bdf",
                    a=a,
                    b=b,
                    h=h,
                    ply_angle=ply_angle,
                )

                AR_g1 = AR if AR > 1 else 1.0 / AR
                _nelems = 3000  # need at least 3000 elements to achieve mesh convergence for this case, ~2000 or less is not converged and has high eigvalue
                min_elem = int(np.sqrt(_nelems / AR_g1))
                max_elem = int(min_elem * AR_g1)
                if AR > 1.0:
                    nx = max_elem
                    ny = min_elem
                else:  # AR < 1.0
                    ny = max_elem
                    nx = min_elem

                print(f"nx = {nx}")
                print(f"ny = {ny}")

                load_factor = 0.03
                flat_plate.generate_tripping_bdf(
                    nx=nx,
                    ny=ny,
                    exx=flat_plate.affine_exx * load_factor,
                    eyy=0.0,
                    exy=0.0,  # flat_plate.affine_exy,
                )

                print(f"xi = {flat_plate.Dstar}")
                epsilon = flat_plate.generalized_poisson
                print(f"eps = {epsilon}")
                # exit()

                # avg_in_plane_loads = flat_plate.run_static_analysis(write_soln=True)
                # exit()

                tacs_eigvals, errors = flat_plate.run_buckling_analysis(
                    sigma=5.0, num_eig=40, write_soln=False
                )  # num_eig = 12 (before) => somehow this change might be affecting soln?

                # compare to exact eigenvalue
                tacs_eigval = tacs_eigvals[0] * load_factor
                if flat_plate.generalized_poisson < 0.25:
                    CF_eigval = 0.476 * flat_plate.xi
                else:
                    CF_eigval = 0.42 * flat_plate.xi
                rel_err = (tacs_eigval - CF_eigval) / CF_eigval
                rel_errs += [rel_err]
                error_0 = errors[0]

                if abs(error_0) < 1e-10 and (0.0 < tacs_eigval < 100.0):
                    data_dict = {
                        # model parameter section
                        "xi": [flat_plate.xi],
                        "gen_eps": [flat_plate.generalized_poisson],
                        "a0/b0": [flat_plate.affine_aspect_ratio],
                        "a/b": [flat_plate.aspect_ratio],
                        "b/h": [flat_plate.slenderness],
                        "kmin": [np.real(tacs_eigval)],
                        "CF_err": [np.real(rel_err)],
                        "error": [np.real(error_0)],
                        # other parameter section
                        "material": [flat_plate.material_name],
                        "ply_angle": [flat_plate.ply_angle],
                        "nx": [nx],
                        "ny": [ny],
                    }
                    inner_ct += 1

                    if comm.rank == 0:
                        df = pd.DataFrame(data_dict)
                        if inner_ct == 1 and not (
                            os.path.exists("stiffener_crippling.csv")
                        ):
                            df.to_csv(
                                "data/stiffener_crippling.csv",
                                mode="w",
                                index=False,
                                header=True,
                            )
                        else:
                            df.to_csv(
                                "data/stiffener_crippling.csv",
                                mode="a",
                                index=False,
                                header=False,
                            )

                    # MPI COMM Barrier in case running with multiple procs
                    comm.Barrier()
