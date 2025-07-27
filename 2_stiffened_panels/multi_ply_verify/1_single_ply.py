import ml_buckling as mlb
from mpi4py import MPI
from tacs import pyTACS, constitutive, elements, utilities, caps2tacs, TACS
import os
import numpy as np
from _src import compute_stiffened_ABD, gen_mesh, rotated_ply_effective_props

"""
we're running directly with TACS objects here and the bdf_file from _stiffened_panel.bdf
"""

comm = MPI.COMM_WORLD

# inputs
# ------------------------

# main (and single ply)
# theta = 0.0
theta = 15.0
# theta = 30.0
# theta = 10.0
# theta = 45.0
# _h_w, _AR = 0.02, 1.0
# _h_w, _AR = 0.04, 1.0
# _h_w, _AR = 0.02, 2.0
_h_w, _AR = 0.027, 1.9
# hw_mult, AR_mult = 0.85, 1.3
hw_mult, AR_mult = 0.87, 1.28

# strain_mult = 10.0 # if need to adjust the strain to solve it better
strain_mult = 1.0

plate_SR = 300.0
# plate_SR = 800.0
# plate_SR = 100.0

# hw_mult = 1e-3

# nelems = 2000
nelems = 1000

# nstiff = 0
nstiff = 1

# stiff_AR = 15.0
stiff_AR = 20.0
# stiff_AR = 80.0

# dependent
h_w = hw_mult * _h_w
AR = _AR * AR_mult
ply_angles = np.deg2rad(np.array([theta]))
ply_fractions = np.array([1.0])
b = 1.0
h = b / plate_SR
t_w = h_w / stiff_AR
a = b * AR
sp = b / (1.0 + nstiff) # for 1 stiffener
# ---------------------------------------

# for composite
E1, E2, G, nu12 = 117.9e9, 9.7e9, 4.8e9, 0.35

# for metal (verified this case matches now..)
# E1, E2, nu12 = 138e9, 138e9, 0.3
# G = E1 / 2.0 / (1 + nu12)

# now compute the nondim params
# -----------------------------

# first the ABD matrix where B = 0
ABD_plate, z_c, E1s, Is = compute_stiffened_ABD(
    E1=E1, E2=E2, G12=G, nu12=nu12,
    plate_thickness=h,
    ply_angles_deg=np.rad2deg(ply_angles),
    ply_fractions=ply_fractions,
    h_w=h_w, t_w=t_w,
    nstiff=nstiff,
)

# now compute the non-dim params now
A = ABD_plate[:3, :3]
D = ABD_plate[3:,3:]
rho0 = a / b * (D[1,1] / D[0,0])**0.25
xi = (D[0,1] + 2.0 * D[-1,-1]) / np.sqrt(D[0,0] * D[1,1])
gamma = E1s * Is / sp / D[0,0]
zeta = (h / b)**2 * A[0,0] / A[-1,-1]
E1p = E1s
if nstiff == 0:
    E1s = 0.0
delta = E1s * h_w * t_w / (E1p * sp * h)

# print(F"{A=}")
print(f"{D=}")
# exit()

print(f"{rho0=:.3e}")
print(f"{xi=:.3e}")
print(f"{gamma=:.3e}")
print(f"{zeta=:.3e}")
print("------------------------------\n\n")

# -----------------------------------

# generate the mesh..
# -----------------------------
stiff_analysis = gen_mesh(comm, a, b, h, h_w, t_w, E1=E1, E2=E2, nu12=nu12, G=G, nstiff=nstiff, 
                          theta=theta, strain_mult=strain_mult, nelems=nelems)

# stiff_analysis.xi_plate
# exit()

# now solve the buckling problem
# -----------------------------------

# Instantiate FEAAssembler
FEAAssembler = pyTACS("_stiffened_panel.bdf", comm=comm)

n_plies = ply_angles.shape[0]


# define an element callback
def elemCallBack(
    dvNum, compID, compDescript, elemDescripts, globalDVs, **kwargs
):

    if "panel" in compDescript:
        plate_thickness = h # for b = 1.0, SR = 100.0, thick = b / SR
    else: # stiffener   
        plate_thickness = t_w # t_w (stiffener thickness)

    ply_thickness = plate_thickness / n_plies

    # have to compute effective rotated E1, E2, nu12, G12 properties
    # so that A16, A26 = 0 in the laminate (otherwise we get axial-shear coupling,
    # and not pure axial response) => so can't train axial modes. This is only way to do that.
    # even if ensure linear static response has no shear load (or deformation can't do both), nonlinear will have shear deformations

    
    ortho_layups = []

    for iply in range(n_plies):
        # get rotated properties
        util = mlb.CompositeMaterialUtility(E1, E2, nu12, G).rotate_ply(theta)
        E1_rot, E2_rot, G12_rot, nu12_rot = util.E11, util.E22, util.G12, util.nu12
        # E1_rot, E2_rot, G12_rot, nu12_rot = rotated_ply_effective_props(E1, E2, G, nu12, theta_rad=ply_angles[iply]) # this method gives wrong nu12 right now (way too large)

        # print(f"{E1=:.3e} {E1_rot=:.3e} {E2=:.3e} {E2_rot=:.3e} {G=:.3e} {G12_rot=:.3e}")
        # print(F"{G=:.3e} {G12_rot=:.3e} {nu12=:.3e} {nu12_rot=:.3e}")
        # exit()
            
        ortho_prop = constitutive.MaterialProperties(
            rho=1.55e3,  # density kg/m^3
            E1=E1_rot,  # Young's modulus in 11 direction (Pa)
            E2=E2_rot,  # Young's modulus in 22 direction (Pa)
            G12=G12_rot,  # in-plane 1-2 shear modulus (Pa)
            # G13=G,  # Transverse 1-3 shear modulus (Pa)
            # G23=G,  # Transverse 2-3 shear modulus (Pa)
            G13=G12_rot,  # Transverse 1-3 shear modulus (Pa)
            G23=G12_rot,  # Transverse 2-3 shear modulus (Pa)
            nu12=nu12_rot,  # 1-2 poisson's ratio
            T1=1648e6,  # Tensile strength in 1 direction (Pa)
            C1=1034e6,  # Compressive strength in 1 direction (Pa)
            T2=64e6,  # Tensile strength in 2 direction (Pa)
            C2=228e6,  # Compressive strength in 2 direction (Pa)
            S12=71e6,  # Shear strength direction (Pa)
        )
        ortho_ply = constitutive.OrthotropicPly(ply_thickness, ortho_prop)
        ortho_layups += [ortho_ply]

    # Create smeared stiffness object based on ply angles/fractions
    # con = constitutive.SmearedCompositeShellConstitutive(

    dtype = np.float64
    con = constitutive.CompositeShellConstitutive(
        ortho_layups,
        np.array([ply_thickness]*n_plies, dtype=dtype),
        ply_angles * 0.0, # since we've adjusted E1, E2 constants, etc.
        tOffset=0.0,
    )

    # Define reference axis to define local 0 deg direction
    # refAxis = np.array([1.0, 0.0, 0.0])
    # refAxis = np.array([np.cos(theta), np.sin(theta), 0.0])
    # transform = elements.ShellRefAxisTransform(refAxis)

    transform = None

    # Pass back the appropriate tacs element object
    elem = elements.Quad4Shell(transform, con)
    return elem

# Set up constitutive objects and elements
FEAAssembler.initialize(elemCallBack)

if not os.path.exists("_oneply"):
    os.mkdir("_oneply")
SP = FEAAssembler.createStaticProblem(
    name="static"
)
SP.solve()
SP.writeSolution(outputDir="_oneply")

bucklingProb = FEAAssembler.createBucklingProblem(
    name="buckle", sigma=5.0, numEigs=10
)
bucklingProb.setOption("printLevel", 2)

# solve and evaluate functions/sensitivities
funcs = {}
funcsSens = {}
bucklingProb.solve()
bucklingProb.evalFunctions(funcs)
bucklingProb.writeSolution(outputDir="_oneply")

eigvals = []
for imode in range(20):
    eigval, eigvec = bucklingProb.getVariables(imode)
    eigvals += [eigval]
    # error = self.bucklingProb.getModalError(imode)

print("\n---------------------")
print(f"single ply, eigval0 = {eigvals[0]:.3e}")
print(f"\t{rho0=:.3e}, {xi=:.3e}")
print(f"\t{gamma=:.3e}, {zeta=:.3e}, {delta=:.3e}")

# now we need to adjust exx for actual nondim buckling load here..
exx = stiff_analysis.affine_exx * strain_mult
buckle_exx = eigvals[0] * exx
A11_eff = A[0,0] - A[0,1]**2 / A[1,1]
buckle_Nxx = buckle_exx * A11_eff

# now nondim the buckling load
buckle_Nxx_nd = buckle_Nxx / np.pi**2 / np.sqrt(D[0,0] * D[1,1]) * b**2 * (1 + delta)
print(f"\t{buckle_Nxx_nd=:.3e}")

# compared to predictions of stiff analysis object (DEBUG / double check)
# print(f"{stiff_analysis}")