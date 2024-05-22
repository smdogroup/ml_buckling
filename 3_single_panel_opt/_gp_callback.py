__all__ = [
    "gp_callback",
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

def gp_callback(
    dvNum, compID, compDescript, elemDescripts, specialDVs, **kwargs
):

    # Create the orthotropic layup
    ortho_prop = constitutive.MaterialProperties(
        rho=1550,
        specific_heat=921.096,
        E1=54e3,
        E2=18e3,
        nu12=0.25,
        G12=9e3,
        G13=9e3,
        G23=9e3,
        Xt=2410.0,
        Xc=1040.0,
        Yt=73.0,
        Yc=173.0,
        S12=71.0,
        alpha=24.0e-6,
        kappa=230.0,
    )
    ortho_ply = constitutive.OrthotropicPly(1e-3, ortho_prop)

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
        1.0, # panel length
        1.0, # stiffener pitch
        100.0, # panel thickness
        10.0, # stiffener height
        100.0, # stiffener thickness
        1.0, # panel width
    ]
    # TBD can add panel and stifener ply fractions to the DVs

    # don't put in any GP models (so using closed-form solutions rn)
    con = constitutive.GPBladeStiffenedShellConstitutive(
        panelPly=ortho_ply,
        stiffenerPly=ortho_ply,
        panelLength=0.5, # choose wrong initial value first to check if it corrects in FUNtoFEM
        stiffenerPitch=0.2,
        panelThick=1.5e-2,
        panelPlyAngles=np.deg2rad(np.array([0.0, -45.0, 45.0, 90.0], dtype=dtype)),
        panelPlyFracs=np.array([44.41, 22.2, 22.2, 11.19], dtype=dtype) / 100.0,
        stiffenerHeight=0.075,
        stiffenerThick=1e-2,
        stiffenerPlyAngles=np.deg2rad(np.array([0.0, -45.0, 45.0, 90.0], dtype=dtype)),
        stiffenerPlyFracs=np.array([44.41, 22.2, 22.2, 11.19], dtype=dtype) / 100.0,
        panelWidth=0.5, # choose wrong initial value first to check if it corrects in FUNtoFEM
        flangeFraction=0.8,
        panelLengthNum=dvNum,
        stiffenerPitchNum=dvNum+1,
        panelThickNum=dvNum+2,
        stiffenerHeightNum=dvNum+3,
        stiffenerThickNum=dvNum+4,
        panelWidthNum=dvNum+5,
    )
    # Set the KS weight really low so that all failure modes make a
    # significant contribution to the failure function derivatives
    con.setKSWeight(20.0)

    con.setStiffenerPitchBounds(0.05, 0.5)
    con.setPanelThicknessBounds(0.002, 0.1)
    con.setStiffenerHeightBounds(0.002, 0.1)
    con.setStiffenerThicknessBounds(0.002, 0.1)

    # --- Create reference axis transform to define the stiffener direction ---
    ref_axis = np.array([1.0, 0.0, 0.0], dtype=dtype)
    transform = elements.ShellRefAxisTransform(ref_axis)

    # --- Create the element object ---
    if elemDescripts[0] == "CQUAD4":
        elem = elements.Quad4Shell(transform, con)
    elif elemDescripts[0] == "CQUAD9":
        elem = elements.Quad9Shell(transform, con)
    elif elemDescripts[0] == "CQUAD16":
        elem = elements.Quad16Shell(transform, con)

    return elem, DVscales
