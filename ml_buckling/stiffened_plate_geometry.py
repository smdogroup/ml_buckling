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
    def area_w(self) -> float:
        return self.t_w * self.h_w
    
    @property
    def area_b(self) -> float:
        return self.w_b * self.t_b
    
    @property
    def area_S(self) -> float:
        return self.area_w + self.area_b
    
    @property
    def area_P(self) -> float:
        return self.b * self.h
    
    @property
    def I_S(self) -> float:
        return self.h_w**3 / 12.0
    
    @property
    def I_P(self) -> float:
        return self.h**3 / 12.0

    @property
    def num_local(self): 
        # number of whole number, 
        # local sections bounded by stiffeners
        return self.num_stiff