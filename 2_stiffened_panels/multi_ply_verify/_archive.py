import ml_buckling as mlb
from mpi4py import MPI
from tacs import pyTACS, constitutive, elements, utilities, caps2tacs, TACS
import os
import numpy as np
from _src import compute_stiffened_ABD, gen_mesh

"""
this old version of the code was trying to compute strain_vec such that in plane loads are only N11 neq 0 and N12 is zero
but when theta > 0 for single-ply, A16 and A26 are neq 0 and lead to axial-shear coupling. In the new script, I just
compute effective rotated E1, E2, G12, nu12 properties for each ply so that the ABD is similar but A16, A26 axial-shear coupling 
is removed and we get pure axial response.. (this is the only way to do that..) I think maybe for the shear panels, I would sometimes
see slightly mixed modes? I can't remember, maybe not. But I could do something similar for that to single vs multi ply comparison.
"""

comm = MPI.COMM_WORLD

# inputs
# ------------------------

# main (and single ply)
# theta = 0.0
theta = 45.0
hw_mult, AR_mult = 1.3, 2.4 

# strain_mult = 10.0 # if need to adjust the strain to solve it better
strain_mult = 1.0

# plate_SR = 300.0
# plate_SR = 800.0
plate_SR = 100.0

# hw_mult = 1e-3

# nelems = 2000
nelems = 1000

nstiff = 0
# nstiff = 1

# dependent
h_w = hw_mult * 0.02
AR = 1.0 * AR_mult
ply_angles = np.deg2rad([theta])
ply_fractions = np.array([1.0])
stiff_AR, b = 20.0, 1.0
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
    ply_angles=np.rad2deg(ply_angles),
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
# print(f"{D=}")

print(f"{rho0=:.3e}")
print(f"{xi=:.3e}")
print(f"{gamma=:.3e}")
print(f"{zeta=:.3e}")
print("------------------------------\n\n")

# exit()

# compute strain ratios such that exx = 1, exy, eyy..
Nvec = np.array([-1.0, 0.0, 0.0]).reshape((3,1))
strain_vec = np.linalg.solve(A, Nvec)
strain_vec /= np.abs(strain_vec[0])
strain_vec = strain_vec[:,0]

print(F"{strain_vec=}")

# double check this produces only N11 strains..
Nvec2 = np.dot(A, strain_vec)
print(f"{Nvec2=}")

# -----------------------------------

# generate the mesh..
# -----------------------------
stiff_analysis = gen_mesh(comm, a, b, h, h_w, t_w, E1=E1, E2=E2, nu12=nu12, G=G, nstiff=nstiff, 
                          theta=theta, strain_mult=strain_mult, strain_vec=strain_vec, nelems=nelems)

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
        plate_thickness = 0.01 # for b = 1.0, SR = 100.0, thick = b / SR
    else: # stiffener   
        plate_thickness = 0.001 # t_w (stiffener thickness)

    ply_thickness = plate_thickness / n_plies
        
    ortho_prop = constitutive.MaterialProperties(
        rho=1.55e3,  # density kg/m^3
        E1=E1,  # Young's modulus in 11 direction (Pa)
        E2=E2,  # Young's modulus in 22 direction (Pa)
        G12=G,  # in-plane 1-2 shear modulus (Pa)
        G13=G,  # Transverse 1-3 shear modulus (Pa)
        G23=G,  # Transverse 2-3 shear modulus (Pa)
        nu12=nu12,  # 1-2 poisson's ratio
        T1=1648e6,  # Tensile strength in 1 direction (Pa)
        C1=1034e6,  # Compressive strength in 1 direction (Pa)
        T2=64e6,  # Tensile strength in 2 direction (Pa)
        C2=228e6,  # Compressive strength in 2 direction (Pa)
        S12=71e6,  # Shear strength direction (Pa)
    )
    ortho_ply = constitutive.OrthotropicPly(ply_thickness, ortho_prop)
    # Create the layup list (one for each angle)
    ortho_layup = [ortho_ply] * n_plies
    # Assign each ply fraction a unique DV
    ply_fraction_dv_nums = np.array(
        [dvNum+i for i in range(n_plies)], dtype=np.intc
    )
    # Create smeared stiffness object based on ply angles/fractions
    # con = constitutive.SmearedCompositeShellConstitutive(
    #     ortho_layup,
    #     plate_thickness,
    #     ply_angles,
    #     ply_fractions,
    #     ply_fraction_dv_nums=ply_fraction_dv_nums,
    # )

    dtype = np.float64
    con = constitutive.CompositeShellConstitutive(
        [ortho_ply],
        np.array([plate_thickness], dtype=dtype),
        np.array([theta], dtype=dtype),
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
# exx = stiff_analysis.affine_exx * strain_mult
# buckle_exx = eigvals[0] * exx
# A11_eff = A[0,0] - A[0,1]**2 / A[1,1]
# buckle_Nxx = buckle_exx * A11_eff

applied_strains = strain_vec * stiff_analysis.affine_exx * strain_mult
buckle_Nxx = np.dot(A, applied_strains)[0]
buckle_Nxx = np.abs(buckle_Nxx)
# print(f"{applied_strains=} {stiff_analysis.affine_exx=} {buckle_Nxx=}")

# now nondim the buckling load
buckle_Nxx_nd = buckle_Nxx / np.pi**2 / np.sqrt(D[0,0] * D[1,1]) * b**2 * (1 + delta)
print(f"\t{buckle_Nxx_nd=:.3e}")

# compared to predictions of stiff analysis object (DEBUG / double check)
# print(f"{stiff_analysis}")