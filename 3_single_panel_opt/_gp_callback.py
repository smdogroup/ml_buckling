__all__ = [
    "gp_callback_generator",
]

"""
Define the TACS constitutive and element objects in the structural analysis.
Also setup the GPBladeConstitutive object to use GP ML models here for the buckling constraints

Author: Sean Engelstad
"""

from tacs import elements, constitutive, TACS
import numpy as np
import ml_buckling as mlb

dtype = TACS.dtype

# ==============================================================================
# Element callback function
# ==============================================================================

# build the GP callback at runtime as you don't know the exact tacs components yet in the general case
# in this single panel simple example - technically you do know all the tacs components
# but let's do it here in the same structure as the real wing case for testing..
def gp_callback_generator(tacs_component_names):
    """
    method to build the gp callback at runtime using the tacs component names
    """

    # build one Axial and Shear GP model to be used for all const objects (no duplication)
    axialGP = constitutive.AxialGP.from_csv(
        csv_file=mlb.axialGP_csv, theta_csv=mlb.axial_theta_csv
    )
    shearGP = constitutive.ShearGP.from_csv(
        csv_file=mlb.shearGP_csv, theta_csv=mlb.shear_theta_csv
    )

    # now build a dictionary of PanelGP objects which manage the GP for each tacs component/panel
    panelGP_dict = constitutive.PanelGPs.component_dict(
        tacs_component_names, axialGP=axialGP, shearGP=shearGP
    )

    def GP_callback(dvNum, compID, compDescript, elemDescripts, specialDVs, **kwargs):

        # make the panelGPs object for the panel that this constitutive object belongs to
        panelGPs = panelGP_dict[compDescript]

        # Create the orthotropic layup
        ortho_prop = constitutive.MaterialProperties(
            rho=1550,
            specific_heat=921.096,
            E1=54e3,  # replace these values with more realistic
            # these values came from unittest script
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
            1.0,  # panel length
            1.0,  # stiffener pitch
            100.0,  # panel thickness
            10.0,  # stiffener height
            100.0,  # stiffener thickness
            1.0,  # panel width
        ]
        # TBD can add panel and stifener ply fractions to the DVs

        con = constitutive.GPBladeStiffenedShellConstitutive(
            panelPly=ortho_ply,
            stiffenerPly=ortho_ply,
            panelLength=0.5,  # choose wrong initial value first to check if it corrects in FUNtoFEM
            stiffenerPitch=0.2,
            panelThick=1.5e-2,
            panelPlyAngles=np.deg2rad(np.array([0.0, -45.0, 45.0, 90.0], dtype=dtype)),
            panelPlyFracs=np.array([44.41, 22.2, 22.2, 11.19], dtype=dtype) / 100.0,
            stiffenerHeight=0.075,
            stiffenerThick=1e-2,
            stiffenerPlyAngles=np.deg2rad(
                np.array([0.0, -45.0, 45.0, 90.0], dtype=dtype)
            ),
            stiffenerPlyFracs=np.array([44.41, 22.2, 22.2, 11.19], dtype=dtype) / 100.0,
            panelWidth=0.5,  # choose wrong initial value first to check if it corrects in FUNtoFEM
            flangeFraction=0.8,
            panelLengthNum=dvNum,
            stiffenerPitchNum=dvNum + 1,
            panelThickNum=dvNum + 2,
            stiffenerHeightNum=dvNum + 3,
            stiffenerThickNum=dvNum + 4,
            panelWidthNum=dvNum + 5,
            panelGPs=panelGPs,
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

    return GP_callback
