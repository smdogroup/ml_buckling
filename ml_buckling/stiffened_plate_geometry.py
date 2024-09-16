__all__ = ["StiffenedPlateGeometry"]

import numpy as np
import math


class StiffenedPlateGeometry:
    def __init__(
        self,
        a,
        b,
        h,
        h_w,  # height of stiffener wall
        t_w,  # thickness of stiffener wall
        w_b=None,  # width of base
        t_b=None,  # thickness of base
        s_p: float = None,  # stiffener pitch
        num_stiff=None,  # num stiffeners
        rib_h=2e-3,  # thickness of rib
    ):
        self.a = a
        self.b = b
        self.h = h

        assert num_stiff is not None or s_p is not None
        self._num_stiff = num_stiff
        self._s_p = s_p

        # not implemented in closed-form yet so ignore this
        assert t_b is None or t_b == 0.0

        self.w_b = w_b if w_b is not None else 0.0
        self.t_b = t_b if t_b is not None else 0.0
        self.h_w = h_w
        self.t_w = t_w

        self.rib_h = rib_h

    @property
    def s_p(self) -> float:
        """stiffener pitch"""
        if self._num_stiff is not None:
            return self.b / self.N
        else:
            return self._s_p

    @property
    def boundary_s_p(self) -> float:
        """stiffener pitch at the boundaries (leftover material spacing), see Quinn example"""
        if self._num_stiff is not None:
            return self.s_p
        else:
            return (self.b / 2.0) % self.s_p

    @property
    def num_stiff(self) -> int:
        if self._num_stiff is not None:
            return self._num_stiff
        else:
            # symmetricly placed stiffeners with extra space at ends
            return 2 * math.ceil(self.b / 2.0 / self.s_p) - 1

    @property
    def N(self) -> int:
        """number of panel sections"""
        return self.num_stiff + 1

    @property
    def area_w(self) -> float:
        return self.t_w * self.h_w

    @property
    def area_b(self) -> float:
        if self.w_b is None or self.t_b is None:
            return 0.0
        else:
            return self.w_b * self.t_b

    @property
    def area_S(self) -> float:
        return self.area_w + self.area_b

    @property
    def area_P(self) -> float:
        return self.b * self.h

    @property
    def I_S(self) -> float:
        return self.h_w ** 3 / 12.0

    @property
    def I_P(self) -> float:
        return self.h ** 3 / 12.0

    @property
    def num_local(self):
        # number of whole number,
        # local sections bounded by stiffeners
        return self.num_stiff + 1

    @property
    def AR(self) -> float:
        return self.a / self.b

    @property
    def SR(self) -> float:
        return self.b / self.h

    @property
    def stiff_AR(self) -> float:
        return self.h_w / self.t_w

    @classmethod
    def copy(cls, geometry):
        return cls(
            a=geometry.a,
            b=geometry.b,
            h=geometry.h,
            num_stiff=geometry.num_stiff,
            w_b=geometry.w_b,
            t_b=geometry.t_b,
            h_w=geometry.h_w,
            t_w=geometry.t_w,
            rib_h=geometry.rib_h,
        )

    @property
    def volume(self) -> float:
        panel_volume = self.a * self.b * self.h
        stiff_volume = self.num_stiff * self.h_w * self.t_w * self.a
        return panel_volume + stiff_volume

    def get_mass(self, density: float):
        return density * self.volume

    def __str__(self):
        mystr = "Stiffened panel geometry object:\n"
        mystr += f"\ta = {self.a}\n"
        mystr += f"\tb = {self.b}\n"
        mystr += f"\th = {self.h}\n"
        mystr += f"\tnum_stiff = {self.num_stiff}\n"
        mystr += f"\tstiffener pitch = {self.s_p}\n"
        mystr += f"\tw_b = {self.w_b}\n"
        mystr += f"\tt_b = {self.t_b}\n"
        mystr += f"\th_w = {self.h_w}\n"
        mystr += f"\tt_w = {self.t_w}\n"
        mystr += f"\tAR = {self.AR}\n"
        mystr += f"\tSR = {self.SR}\n"
        mystr += f"\tstiff_AR = {self.stiff_AR}\n"
        return mystr
