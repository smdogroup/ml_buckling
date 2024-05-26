
__all__ = ["Symbol", "Sin", "Cos", "Float", "Zero", "One"]

"""
AUTHOR: Sean Engelstad
NOTE: the purpose of this symbolic manipulation package is that I need certain features of symbolic manipulation
for the shear closed-form solution that Mathematica doesn't seem to be doing that well. I am able to do TrigReduce and 
expand a product of a bunch of sines into a sum of one factor sines and cosines. However, mathematica is doing the analytic integrals
using pi rounded (it does this when the runtime exceeds a certain amount I think). So I basically need to write my own
symbolic manipulator, and do these analytic integrals myself to get around this behavior. Lol.
"""

def float_eq(num1, num2, rtol=1e-10):
    if num1 == 0.0 or num2 == 0.0:
        return num1 == num2
    else:
        return abs(num1-num2)/abs(num1) < rtol

class Base:
    def __add__(self, obj):
        if self == Zero():
            return obj
        elif obj == Zero():
            return self
        else:
            return Add(self, obj)
    
    def __sub__(self, obj):
        if self == Zero():
            return obj * Float(-1.0)
        elif obj == Zero():
            return self
        else:
            return Subtract(self, obj)

    def __mul__(self, obj):
        if self == Zero() or obj == Zero():
            return Zero()
        elif self == One():
            return obj
        elif obj == One():
            return self
        else:
            return Multiply(self, obj)
    
    def __truediv__(self, obj):
        if self == Zero():
            return Zero()
        elif obj == Zero():
            raise AssertionError("Divide by Zero error")
        elif obj == One():
            return self
        else:
            return Divide(self, obj)

class Symbol(Base):
    def __init__(self, name, float=1.0, exponent=1, float_type="f"):
        if isinstance(name, Symbol):
            self.name = name.name
        elif isinstance(name, str):
            self.name = name
        else:
            raise AssertionError("Name put in Symbol constructor is not correct type")
        self.name = name
        self.float = float
        self.exponent = exponent
        assert float_type in ["f", "e"]
        self.float_type = float_type

    @property
    def operator_type(self):
        return self.get_operator_type(self.operators)

    def __add__(self, sym):
        if isinstance(sym, Symbol) and self.name == sym.name and self.exponent == sym.exponent:
            return Symbol(self.name, float=self.float+sym.float)
        else:
            return Base.__add__(self,sym)
        
    def __sub__(self, sym):
        # simplify adding or subtracting zero terms (identity)
        if isinstance(sym, Symbol) and self.name == sym.name and self.exponent == sym.exponent:
            return Symbol(self.name, float=self.float-sym.float, exponent=self.exponent)
        else:
            return Base.__sub__(self,sym)

    def __mul__(self, sym):
        # several different cases of float * symbol (for auto simplification)
        if isinstance(self,Float) and isinstance(sym,Float):
            return Float(self.float * sym.float)
        elif isinstance(self,Float):
            new_self = Symbol(sym.name, float=self.float, exponent=0.0)
            return new_self.__mul__(sym)
        elif isinstance(sym,Float):
            new_sym = Symbol(sym.name, float=sym.float, exponent=0.0)
            return self.__mul__(new_sym)
        elif isinstance(sym, Symbol) and self.name == sym.name:
            return Symbol(self.name, float=self.float*sym.float, 
                    exponent=self.exponent + sym.exponent)
        else:
            return Base.__mul__(self,sym)
    
    def __truediv__(self, sym):
        # several different cases of float * symbol (for auto simplification)
        if isinstance(self,Float) and isinstance(sym,Float):
            return Float(self.float / sym.float)
        elif isinstance(self,Float):
            new_self = Symbol(sym.name, float=self.float, exponent=0.0)
            return new_self.__truediv__(sym)
        elif isinstance(sym,Float):
            new_sym = Symbol(sym.name, float=sym.float, exponent=0.0)
            return self.__truediv__(new_sym)
        elif isinstance(self, Symbol) and self.name == sym.name:
            if self.exponent == sym.exponent:
                return Float(self.float / sym.float)
            else:
                return Symbol(self.name, float=self.float/sym.float, 
                    exponent=self.exponent - sym.exponent)
        else:
            return Base.__truediv__(self,sym)
        
    def derivative(self, sym, order=1):
        assert isinstance(order, int) and order >= 1
        if isinstance(sym, str):
            sym = Symbol(sym)
        if self.name != sym.name:
            return Zero()
        
        first_deriv = Symbol(
            name=self.name,
            float=self.float * self.exponent,
            exponent=self.exponent-1,
        )
        # recursively call higher order derivatives
        if order > 1:
            return first_deriv.derivative(sym, order=order-1)
        else:
            return first_deriv
    
    @property
    def name_str(self):
        if self.exponent == 1:
            return f"{self.name}"
        else:
            return f"{self.name}^{self.exponent}"

    def __str__(self):
        if self.float == 1.0:
            return self.name_str
        elif self.float_type == "f":
            return f"{self.float:.4f}*" + self.name_str
        elif self.float_type == "e":
            return f"{self.float:.4e}*" + self.name_str

    def __eq__(self, obj):
        if not isinstance(obj,Symbol):
            return False
        else:
            return self.name == obj.name and float_eq(self.float, obj.float) and float_eq(self.exponent, obj.exponent)


def add_sub_type(var):
    return isinstance(var, Symbol) or isinstance(var, Add) or isinstance(var, Subtract)

def mul_div_type(var):
    return isinstance(var, Symbol) or isinstance(var, Multiply) or isinstance(var, Divide)

class Float(Symbol):
    def __init__(self, float):
        super(Float,self).__init__(
            name="",
            float=float,
            exponent=0.0,
        )

    def derivative(self, sym, order=1):
        return Zero()

    def __eq__(self, obj):
        if isinstance(obj, Float):
            return self.float == obj.float
        else:
            return False

class Zero(Float):
    def __init__(self):
        super(Zero,self).__init__(
            float=0.0
        )

class One(Float):
    def __init__(self):
        super(One,self).__init__(
            float=1.0
        )

class Binary(Base):
    def __Init__(self, left, right):
        self.left = left
        self.right = right

    @property
    def type_method(self):
        return None
    
    @property
    def op(self):
        return None
    
    @property
    def parentheses(self):
        if self.type_method(self.left) and self.type_method(self.right):
            return (False,False)
        else:
            return (not( isinstance(self.left, Symbol) or isinstance(self.left, Unary)),
                    not( isinstance(self.right, Symbol) or isinstance(self.right, Unary)))
        
    def __str__(self):
        parenth = self.parentheses
        left_string = f"({self.left})" if parenth[0] else f"{self.left}"
        right_string = f"({self.right})" if parenth[1] else f"{self.right}"
        return f"{left_string}{self.op}{right_string}"

class Add(Binary):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    @property
    def type_method(self):
        return add_sub_type
    
    @property
    def op(self):
        return "+"
    
    def derivative(self, sym, order=1):
        return self.left.derivative(sym,order) + self.right.derivative(sym,order)
    
    def __eq__(self, obj):
        if not isinstance(obj, Add):
            return False
        else:
            return self.left == obj.left and self.right == obj.right
    
class Subtract(Binary):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    @property
    def type_method(self):
        return add_sub_type
    
    @property
    def op(self):
        return "-"

    def derivative(self, sym, order=1):
        return self.left.derivative(sym,order) - self.right.derivative(sym,order)
    
    def __eq__(self, obj):
        if not isinstance(obj, Subtract):
            return False
        else:
            return self.left == obj.left and self.right == obj.right
    
class Multiply(Binary):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    @property
    def type_method(self):
        return mul_div_type
    
    @property
    def op(self):
        return "*"
    
    def derivative(self, sym, order=1):
        # product rule
        left = self.left.derivative(sym,order) * self.right
        right = self.left * self.right.derivative(sym,order)
        return left + right
    
    def __eq__(self, obj):
        if not isinstance(obj, Multiply):
            return False
        else:
            return self.left == obj.left and self.right == obj.right
    
class Divide(Binary):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    @property
    def type_method(self):
        return mul_div_type
    
    @property
    def op(self):
        return "/"
    
    def derivative(self, sym, order=1):
        # quotient rule
        left = self.left.derivative(sym,order).__mul__(self.right)
        left = self.left.derivative(sym,order) * self.right
        temp = self.left * self.right.derivative(sym,order)
        right = temp / self.right / self.right
        return left - right
    
    def __eq__(self, obj):
        if not isinstance(obj, Divide):
            return False
        else:
            return self.left == obj.left and self.right == obj.right
    
class Unary(Base):
    def __init__(self, arg):
        self.arg = arg

    @property
    def unary_name(self):
        return None
    
    def __str__(self):
        return f"{self.unary_name}({self.arg})"    

class Sin(Unary):
    def __init__(self, arg):
        self.arg = arg

    @property
    def unary_name(self):
        return "sin"
    
    def derivative(self, sym, order=1):
        # chain rule on sine
        return Cos(self.arg) * self.arg.derivative(sym,order)

class Cos(Unary):
    def __init__(self, arg):
        self.arg = arg

    @property
    def unary_name(self):
        return "cos"
    
    def derivative(self, sym, order=1):
        # chain rule on sine
        return Sin(self.arg) * self.arg.derivative(sym,order) * Float(-1.0)