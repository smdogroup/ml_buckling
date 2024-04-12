__all__ = ["CompositeMaterial"]

from .composite_material_utility import CompositeMaterialUtility
import numpy as np


class CompositeMaterial:
    def __init__(
        self,
        E11,
        nu12,
        E22=None,
        G12=None,
        _G23=None,
        _G13=None,
        ply_angles=None,
        material_name=None,
        ply_fractions=None,
        symmetric=True,
        ref_axis=None,
    ):
        self.E11 = E11
        self.nu12 = nu12
        self._E22 = E22
        self._G12 = G12
        self._G23 = _G23
        self._G13 = _G13
        self._ply_angles = ply_angles
        self._ply_fractions = ply_fractions
        self.material_name = material_name
        self.symmetric = symmetric
        self.ref_axis = np.array(ref_axis)

    @property
    def num_plies(self) -> int:
        assert len(self.ply_angles) == len(self.ply_fractions)
        assert sum(self.ply_fractions) == 1.0
        return len(self.ply_angles)

    @property
    def ply_angles(self) -> list:
        if self.symmetric:
            return self._ply_angles + self._ply_angles[::-1]
        else:
            return self._ply_angles

    @property
    def rad_ply_angles(self) -> list:
        return np.deg2rad(self.ply_angles)

    @property
    def ply_fractions(self) -> list:
        if self.symmetric:
            half_fractions = [_ * 0.5 for _ in self._ply_fractions]
            return half_fractions + half_fractions[::-1]
        else:
            return self._ply_fractions

    def get_ply_thicknesses(self, thickness):
        return [thickness * frac for frac in self.ply_fractions]

    @property
    def nu21(self) -> float:
        """reversed 12 Poisson's ratio"""
        return self.nu12 * self.E22 / self.E11

    @property
    def E22(self) -> float:
        if self._E22 is None:
            return self.E11
        else:
            return self._E22

    @property
    def G12(self) -> float:
        if self._G12 is None:
            return self.E11 / 2.0 / (1 + self.nu12)
        else:
            return self._G12

    @property
    def Q_array(self):
        _array = np.zeros((4,))
        for iply, ply_angle in enumerate(self.ply_angles):
            ply_frac = self.ply_fractions[iply]
            util = CompositeMaterialUtility(
                E11=self.E11, E22=self.E22, nu12=self.nu12, G12=self.G12
            ).rotate_ply(ply_angle)

            nu_denom = 1 - util.nu12 * util.nu21
            _array[0] += util.E11 / nu_denom * ply_frac  # Q11
            _array[1] += util.E22 * util.nu12 / nu_denom * ply_frac  # Q12
            _array[2] += util.E22 / nu_denom * ply_frac  # Q22
            _array[3] += util.G12 * ply_frac  # Q66
        return _array

    @property
    def Q11(self) -> float:
        return self.Q_array[0]

    @property
    def Q12(self) -> float:
        return self.Q_array[1]

    @property
    def Q22(self) -> float:
        return self.Q_array[2]

    @property
    def Q66(self) -> float:
        return self.Q_array[3]

    @property
    def E_eff(self) -> float:
        """effective modulus"""
        return self.Q11 - self.Q12 ** 2 / self.Q66

    @classmethod
    def get_materials(cls):
        return [
            cls.solvay5320,
            cls.solvayMTM45,
            cls.torayBT250E,
            cls.hexcelIM7,
            cls.victrexAE,
        ]

    @classmethod
    def get_material_from_str(cls, mat_name: str):
        method_names = [_.__qualname__ for _ in cls.get_materials()]
        materials = cls.get_materials()
        _method = None
        for i, method_name in enumerate(method_names):
            if mat_name in method_name:
                _method = materials[i]
        assert _method is not None
        return _method

    # MATERIALS CLASS METHODS
    # -----------------------------------------------------------

    # NIAR composite materials

    @classmethod
    def solvay5320(cls, ply_angles=[0], ply_fractions=[1], ref_axis=None):
        """
        NIAR dataset - Solvay 5320-1 material (thermoset)
        Fiber: T650 unitape, Resin: Cycom 5320-1
        Room Temperature Dry (RTD) mean properties shown below
        units in Pa, ND
        """

        return cls(
            material_name="solvay5320",
            ply_angles=ply_angles,
            ply_fractions=ply_fractions,
            E11=138.461e9,
            E22=9.177e9,
            nu12=0.326,
            G12=4.957e9,
            ref_axis=ref_axis,
        )

    @classmethod
    def solvayMTM45(cls, ply_angles=[0], ply_fractions=[1], ref_axis=None):
        """
        NIAR dataset - Solvay MTM45 material (thermoset)
        Style: 12K AS4 Unidirectional
        Room Temperature Dry (RTD) mean properties shown below
        units in Pa, ND
        """

        return cls(
            material_name="solvayMTM45",
            ply_angles=ply_angles,
            ply_fractions=ply_fractions,
            E11=129.5e9,
            E22=7.936e9,
            nu12=0.313,
            G12=4.764e9,
            ref_axis=ref_axis,
        )

    @classmethod
    def torayBT250E(cls, ply_angles=[0], ply_fractions=[1], ref_axis=None):
        """
        NIAR dataset - Toray (formerly Tencate) BT250E-6 S2 Unitape Gr 284 material (thermoset)
        Room Temperature Dry (RTD) mean properties shown below
        units in Pa, NDs
        """

        return cls(
            material_name="torayBT250E",
            ply_angles=ply_angles,
            ply_fractions=ply_fractions,
            E11=44.74e9,
            E22=11.36e9,
            nu12=0.278,
            G12=3.77e9,
            ref_axis=ref_axis,
        )

    @classmethod
    def victrexAE(cls, ply_angles=[0], ply_fractions=[1], ref_axis=None):
        """
        NIAR dataset - Victrex AE 250 LMPAEK (thermoplastic)
        Room Temperature Dry (RTD) mean properties shown below
        units in Pa, ND
        """

        return cls(
            material_name="victrexAE",
            ply_angles=ply_angles,
            ply_fractions=ply_fractions,
            E11=131.69e9,
            E22=9.694e9,
            nu12=0.3192,
            G12=4.524e9,
            ref_axis=ref_axis,
        )

    @classmethod
    def hexcelIM7(cls, ply_angles=[0], ply_fractions=[1], ref_axis=None):
        """
        NIAR dataset - Hexcel 8552 IM7 Unidirectional Prepreg (thermoset)
        Room Temperature Dry (RTD) mean properties shown below
        units in Pa, ND
        """

        return cls(
            material_name="torayBT250E",
            ply_angles=ply_angles,
            ply_fractions=ply_fractions,
            E11=158.51e9,
            nu12=0.316,
            E22=8.96e9,
            G12=4.688e9,
            ref_axis=ref_axis,
        )

    def __str__(self):
        mystr = "Composite material object\n"
        mystr += f"\tE11 = {self.E11}\n"
        mystr += f"\tnu12 = {self.nu12}\n"
        mystr += f"\t_E22 = {self._E22}\n"
        mystr += f"\t_G12 = {self._G12}\n"
        mystr += f"\t_G23 = {self._G23}\n"
        mystr += f"\t_G13 = {self._G13}\n"
        mystr += f"\t_ply_angles = {self._ply_angles}\n"
        mystr += f"\t_ply_fractions = {self._ply_fractions}\n"
        mystr += f"\tmaterial_name = {self.material_name}\n"
        mystr += f"\tsymmetric = {self.symmetric}\n"
        mystr += f"\tref_axis = {self.ref_axis}\n"
        return mystr
