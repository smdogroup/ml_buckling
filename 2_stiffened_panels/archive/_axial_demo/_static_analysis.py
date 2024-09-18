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
    b=0.1,
    h=5e-3,
    num_stiff=3,
    h_w=1e-2,
    t_w=1e-3,  # if the wall thickness is too low => stiffener crimping failure happens
)

material = mlb.CompositeMaterial.solvay5320(
    ply_angles=[0],
    ply_fractions=[1.0],
    ref_axis=[1, 0, 0],
)

stiff_analysis = mlb.StiffenedPlateAnalysis(
    comm=comm,
    geometry=geometry,
    stiffener_material=material,
    plate_material=material,
)

stiff_analysis.pre_analysis(
    nx_plate=30,
    ny_plate=30,
    nz_stiff=20,
    exx=stiff_analysis.affine_exx,
    exy=0.0,
    clamped=False,
    _make_rbe=False,
)
avg_stresses = stiff_analysis.run_static_analysis(write_soln=True)
stiff_analysis.post_analysis()

print(f"avg_stresses = {avg_stresses}")
