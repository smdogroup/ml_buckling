__all__ = [
    "closed_form_callback",
]

"""
Define the TACS constitutive and element objects in the structural analysis.
Also setup the GPBladeConstitutive object to use closed-form solutions not the GP models in this

Author: Sean Engelstad
"""

from tacs import elements, constitutive, TACS
import numpy as np

dtype = TACS.dtype


# ==============================================================================
# Element callback function
# ==============================================================================


def closed_form_callback(
    dvNum, compID, compDescript, elemDescripts, specialDVs, **kwargs
):

    # Aluminum alloy Al-MS89
    E = 90e9
    nu = 0.3
    G = E / 2.0 / (1+nu)
    strength = 420e6

    # Create the orthotropic layup
    ortho_prop = constitutive.MaterialProperties(
        rho=2.69e3,  # density kg/m^3
        E1=E,  # Young's modulus in 11 direction (Pa)
        E2=E,  # Young's modulus in 22 direction (Pa)
        G12=G,  # in-plane 1-2 shear modulus (Pa)
        G13=G,  # Transverse 1-3 shear modulus (Pa)
        G23=G,  # Transverse 2-3 shear modulus (Pa)
        nu12=nu,  # 1-2 poisson's ratio
        T1=strength,  # Tensile strength in 1 direction (Pa)
        C1=strength,  # Compressive strength in 1 direction (Pa)
        T2=strength,  # Tensile strength in 2 direction (Pa)
        C2=strength,  # Compressive strength in 2 direction (Pa)
        S12=0.6 * strength,  # Shear strength direction (Pa)
    )
    ortho_ply = constitutive.OrthotropicPly(1.0, ortho_prop)

    # case by case initial ply angles
    if "OML" in compDescript:
        plyAngles = np.deg2rad(np.array([0.0], dtype=dtype))
        panelPlyFractions = np.array([1.0], dtype=dtype)
        refAxis = np.array([0.34968083, 0.93686889, 0.0])
    else:
        plyAngles = np.deg2rad(np.array([0.0], dtype=dtype))
        panelPlyFractions = np.array([1.0], dtype=dtype)
        refAxis = np.array([0.0, 0.0, 1.0])

    # The ordering of the DVs used by the GPBladeStiffenedShell model is:
    # - panel length
    # - stiffener pitch
    # - panel thickness
    # - panel ply fractions (not used in this case)
    # - stiffener height
    # - stiffener thickness
    # - stiffener ply fractions (not used in this case)
    # - panel width

    # create the design variable scales array
    DVscales = [
        1.0,  # panel length
        1.0,  # stiffener pitch
        100.0,  # panel thickness
        10.0,  # stiffener height
        100.0,  # stiffener thickness
        1.0,  # panel width
    ]
    # TBD can add panel and stifener ply fractions to the DVs

    # don't put in any GP models (so using closed-form solutions rn)
    con = constitutive.GPBladeStiffenedShellConstitutive(
        panelPly=ortho_ply,
        stiffenerPly=ortho_ply,
        panelLength=0.5,  # choose wrong initial value first to check if it corrects in FUNtoFEM
        stiffenerPitch=0.2,
        panelThick=1.5e-2,
        panelPlyAngles=plyAngles,
        panelPlyFracs=panelPlyFractions,
        stiffenerHeight=0.075,
        stiffenerThick=1e-2,
        stiffenerPlyAngles=plyAngles,
        stiffenerPlyFracs=np.array([1.0], dtype=dtype),
        panelWidth=0.5,  # choose wrong initial value first to check if it corrects in FUNtoFEM
        flangeFraction=0.0, #0.8
        panelLengthNum=dvNum,
        stiffenerPitchNum=dvNum + 1,
        panelThickNum=dvNum + 2,
        stiffenerHeightNum=dvNum + 3,
        stiffenerThickNum=dvNum + 4,
        panelWidthNum=dvNum + 5,
    )
    # Set the KS weight really low so that all failure modes make a
    # significant contribution to the failure function derivatives
    con.setKSWeight(100.0)
    # con.setWriteDVMode(2)

    con.setStiffenerPitchBounds(0.05, 0.5)
    con.setPanelThicknessBounds(0.002, 0.1)
    con.setStiffenerHeightBounds(0.002, 0.1)
    con.setStiffenerThicknessBounds(0.002, 0.1)

    # --- Create reference axis transform to define the stiffener direction ---
    transform = elements.ShellRefAxisTransform(refAxis)

    # --- Create the element object ---
    if elemDescripts[0] == "CQUAD4":
        elem = elements.Quad4Shell(transform, con)
    elif elemDescripts[0] == "CQUAD9":
        elem = elements.Quad9Shell(transform, con)
    elif elemDescripts[0] == "CQUAD16":
        elem = elements.Quad16Shell(transform, con)

    return elem, DVscales
