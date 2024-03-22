__all__ = ["StiffenedPlateGeometry"]

class StiffenedPlateGeometry:
    def __init__(
        self,
        a,
        b,
        h,
        s_p, # stiffener pitch
        w_b, # width of base
        t_b, # thickness of base
        t_s, # thickness of stiffener
        h_s, # height of stiffener
    ):
        self.a = a
        self.b = b
        self.h = h
        self.s_p = s_p
        self.w_b = w_b
        self.t_b = t_b
        self.t_s = t_s
        self.h_s = h_s