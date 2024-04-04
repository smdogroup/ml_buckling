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

comm = MPI.COMM_WORLD

h = 1.0
SR = 27.1494

# AR_vec = [11.7210]
AR_vec = [0.2, 1, 2,4,6,8,10]
rel_errs = []
for AR in AR_vec:

    b = h * SR
    a = b * AR

    flat_plate = mlb.UnstiffenedPlateAnalysis.solvay5320(
        comm=comm,
        bdf_file="plate.bdf",
        a=a,
        b=b,
        h=h,
        ply_angle=30.36,
    )

    AR_g1 = AR if AR > 1 else 1.0/AR
    _nelems = 3000 # need at least 3000 elements to achieve mesh convergence for this case, ~2000 or less is not converged and has high eigvalue
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
        exy=0.0, # flat_plate.affine_exy,
    )

    print(f"xi = {flat_plate.Dstar}")
    epsilon = flat_plate.generalized_poisson
    print(f"eps = {epsilon}")
    # exit()

    # avg_in_plane_loads = flat_plate.run_static_analysis(write_soln=True)
    # exit()

    tacs_eigvals, errors = flat_plate.run_buckling_analysis(
        sigma=5.0, num_eig=40, write_soln=False
    ) # num_eig = 12 (before) => somehow this change might be affecting soln?

    # compare to exact eigenvalue
    tacs_eigval = tacs_eigvals[0] * load_factor
    CF_eigval = 0.476 * flat_plate.xi
    rel_err = (tacs_eigval - CF_eigval) / CF_eigval
    rel_errs += [rel_err]

    print(f"tacs eigval = {tacs_eigval}")
    print(f"CF eigval = {CF_eigval}")
    print(f"rel err = {rel_err}")

print(f"ARs = {AR_vec}")
print(f"rel errors = {rel_errs}")