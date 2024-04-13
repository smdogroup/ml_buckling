__all__ = ["StiffenedPlateGeometry"]

import numpy as np


class StiffenedPlateGeometry:
    def __init__(
        self,
        a,
        b,
        h,
        num_stiff,  # num stiffeners
        h_w,  # height of stiffener wall
        t_w,  # thickness of stiffener wall
        w_b=None,  # width of base
        t_b=None,  # thickness of base
        rib_h=2e-3,  # thickness of rib
    ):
        self.a = a
        self.b = b
        self.h = h
        self.num_stiff = num_stiff
        self.N = num_stiff + 1

        # not implemented in closed-form yet so ignore this
        assert t_b is None or t_b is 0.0

        self.w_b = w_b if w_b is not None else 0.0
        self.t_b = t_b if t_b is not None else 0.0
        self.h_w = h_w
        self.t_w = t_w

        self.rib_h = rib_h

    @property
    def s_p(self) -> float:
        """stiffener pitch"""
        return self.b / self.N

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

    def __str__(self):
        mystr = "Stiffened panel geometry object:\n"
        mystr += f"\ta = {self.a}\n"
        mystr += f"\tb = {self.b}\n"
        mystr += f"\th = {self.h}\n"
        mystr += f"\tnum_stiff = {self.num_stiff}\n"
        mystr += f"\tspar pitch = {self.s_p}\n"
        mystr += f"\tw_b = {self.w_b}\n"
        mystr += f"\tt_b = {self.t_b}\n"
        mystr += f"\th_w = {self.h_w}\n"
        mystr += f"\tt_w = {self.t_w}\n"
        mystr += f"\tAR = {self.AR}\n"
        mystr += f"\tSR = {self.SR}\n"
        mystr += f"\tstiff_AR = {self.stiff_AR}\n"
        return mystr
