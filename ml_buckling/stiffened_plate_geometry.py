__all__ = ["StiffenedPlateGeometry"]

import numpy as np

class StiffenedPlateGeometry:
    def __init__(
        self,
        a,
        b,
        h,
        num_stiff, # num stiffeners
        w_b, # width of base
        t_b, # thickness of base
        h_w, # height of stiffener wall
        t_w, # thickness of stiffener wall
    ):
        self.a = a
        self.b = b
        self.h = h
        self.num_stiff = num_stiff
        
        self.w_b = w_b
        self.t_b = t_b
        self.h_w = h_w
        self.t_w = t_w

    @property
    def s_p(self) -> float:
        """stiffener pitch"""
        return self.a / self.num_stiff

    @property
    def num_local(self): 
        # number of whole number, 
        # local sections bounded by stiffeners
        return self.num_stiff