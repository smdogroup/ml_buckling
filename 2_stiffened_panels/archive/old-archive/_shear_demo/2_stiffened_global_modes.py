"""
Sean Engelstad
: 
Demo analysis of a stiffened panel
NOTE : copy u*iHat+v*jHat+w*kHat for paraview
"""

import ml_buckling as mlb
from mpi4py import MPI

comm = MPI.COMM_WORLD

# I think what happens when h becomes very low for large a,b the norm of K is too small and that affects the soln
# to prevent low thickness problems => just make overall plate size smaller

geometry = mlb.StiffenedPlateGeometry(
    a=0.1,
    b=0.2,  # 0.1
    h=5e-3,
    num_stiff=3,
    h_w=1e-4,
    t_w=1e-3,  # if the wall thickness is too low => stiffener crimping failure happens
)

material = mlb.CompositeMaterial.solvay5320(
    ply_angles=[38.0], ply_fractions=[1.0], ref_axis=[1, 0, 0]
)

stiff_analysis = mlb.StiffenedPlateAnalysis(
    comm=comm,
    geometry=geometry,
    stiffener_material=material,
    plate_material=material,
)

nx = 40
ny = 20
nz = 10

stiff_analysis.pre_analysis(
    exx=0.0,
    exy=stiff_analysis.affine_exy,
    clamped=False,
    nx_plate=int(nx),
    ny_plate=int(ny),
    nz_stiff=int(nz),
    # global_mesh_size=global_mesh_size,
    # edge_pt_min=5,
    # edge_pt_max=50,
    _make_rbe=True,  # True
)

stresses = stiff_analysis.run_static_analysis(write_soln=True)

if comm.rank == 0:
    print(f"stresses = {stresses}")
    print(stiff_analysis)

tacs_eigvals, errors = stiff_analysis.run_buckling_analysis(
    sigma=5.0, num_eig=20, write_soln=True
)
stiff_analysis.post_analysis()

global_lambda_star = stiff_analysis.min_global_mode_eigenvalue

if comm.rank == 0:
    print(f"global lam star = {global_lambda_star}")
    print(f"tacs eigvals = {tacs_eigvals}")
    print(f"errors = {errors}")
