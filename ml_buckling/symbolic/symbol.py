
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
    @property
    def is_add_sub(self):
        return isinstance(self,Symbol) or isinstance(self,Add) or isinstance(self,Subtract) or isinstance(self,AddSubGroup)
    
    @property
    def is_mul_div(self):
        return isinstance(self,Symbol) or isinstance(self,MultDivGroup) or isinstance(self,Multiply) or isinstance(self,Divide)

    def __add__(self, obj):
        if self == Zero():
            return obj
        elif obj == Zero():
            return self
        elif self.is_add_sub and obj.is_add_sub:
            self_group = AddSubGroup.cast(self)
            obj_group = AddSubGroup.cast(obj)
            return AddSubGroup(
                args=self_group.args + obj_group.args,
                operators=self_group.operators + ["+"] + obj_group.operators,
            )
        else:
            return Add(self, obj)
    
    def __sub__(self, obj):
        if self == Zero():
            return obj * Float(-1.0)
        elif obj == Zero():
            return self
        elif self.is_add_sub and obj.is_add_sub:
            self_group = AddSubGroup.cast(self)
            obj_group = AddSubGroup.cast(obj)
            return AddSubGroup(
                args=self_group.args + obj_group.args,
                operators=self_group.operators + ["-"] + obj_group.operators,
            )
        else:
            return Subtract(self, obj)

    def __mul__(self, obj):
        if self == Zero() or obj == Zero():
            return Zero()
        elif self == One():
            return obj
        elif obj == One():
            return self
        elif self.is_mul_div and obj.is_mul_div:
            self_group = MultDivGroup.cast(self)
            obj_group = MultDivGroup.cast(obj)
            return MultDivGroup(
                args=self_group.args + obj_group.args,
                operators=self_group.operators + ["*"] + obj_group.operators,
            )
        else:
            return Multiply(self, obj)
    
    def __truediv__(self, obj):
        if self == Zero():
            return Zero()
        elif obj == Zero():
            raise AssertionError("Divide by Zero error")
        elif obj == One():
            return self
        elif self.is_mul_div and obj.is_mul_div:
            self_group = MultDivGroup.cast(self)
            obj_group = MultDivGroup.cast(obj)
            return MultDivGroup(
                args=self_group.args + obj_group.args,
                operators=self_group.operators + ["/"] + obj_group.operators,
            )
        else:
            return Divide(self, obj)
        
    @property
    def class_order(self):
        """order of operations"""
        return -1

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
    def class_order(self):
        """order of operations"""
        return 0

    @property
    def operator_type(self):
        return self.get_operator_type(self.operators)
    
    @property
    def is_float(self) -> bool:
        return isinstance(self,Float) or self.exponent == 0.0
    
    @property
    def is_zero(self) -> bool:
        return isinstance(self,Zero) or self.float == 0.0

    def matching_name(self, sym):
        if isinstance(sym,Symbol):
            if self.is_float or sym.is_float:
                return True
            else:
                return self.name == sym.name
        else:
            return False
        
    def get_name(self, sym):
        """get the matching name for simplification"""
        if self.is_float:
            return sym.name
        else:
            return self.name
            
    def matching_exponent(self, sym):
        if isinstance(sym,Symbol):
            return self.exponent == sym.exponent
        else:
            return False
            
    def matching_sym(self, sym):
        if isinstance(sym,Symbol):
            return self.matching_exponent(sym) and self.matching_name(sym)
        else:
            return False
        
    @property
    def simplify(self):
        """change to One or Zero or Float if it is equivalent"""
        if self.is_zero:
            return Zero()
        elif self == One():
            return One()
        elif self.exponent == 0.0:
            return Float(self.float)
        else:
            return self

    def __add__(self, sym):
        if self.matching_name(sym) and self.matching_exponent(sym):
            return Symbol(self.get_name(sym), float=self.float+sym.float, exponent=self.exponent).simplify
        else:
            return Base.__add__(self,sym)
        
    def __sub__(self, sym):
        # simplify adding or subtracting zero terms (identity)
        if self.matching_name(sym) and self.matching_exponent(sym):
            return Symbol(self.get_name(sym), float=self.float-sym.float, exponent=self.exponent).simplify
        else:
            return Base.__sub__(self,sym)

    def __mul__(self, sym):
        # several different cases of float * symbol (for auto simplification)
        if self.matching_name(sym):
            return Symbol(self.get_name(sym), float=self.float*sym.float, exponent=self.exponent+sym.exponent).simplify
        else:
            return Base.__mul__(self,sym)
    
    def __truediv__(self, sym):
        # several different cases of float * symbol (for auto simplification)
        if self.matching_name(sym):
            return Symbol(self.get_name(sym), float=self.float/sym.float, exponent=self.exponent-sym.exponent).simplify
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
            return first_deriv.simplify
    
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
            
    def __str__(self):
        return str(self.float)

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
    def __init__(self, left, right):
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
    
    @property
    def class_order(self):
        """order of operations"""
        return 1
    
    def derivative(self, sym, order=1):
        return self.left.derivative(sym,order) + self.right.derivative(sym,order)
    
    def __eq__(self, obj):
        if not isinstance(obj, Add):
            return False
        # commutative equality
        elif self.left == obj.left and self.right == obj.right:
            return True
        elif self.left == obj.right and self.right == obj.left:
            return True
        else:
            return False
        
    @property
    def simplify(self):
        if self.left.class_order > self.right.class_order:
            return self.right.simplify + self.left.simplify
        else:
            return self.left.simplify + self.right.simplify
    
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
    
    @property
    def class_order(self):
        """order of operations"""
        return 1

    def derivative(self, sym, order=1):
        return self.left.derivative(sym,order) - self.right.derivative(sym,order)
    
    def __eq__(self, obj):
        if not isinstance(obj, Subtract):
            return False
        else:
            return self.left == obj.left and self.right == obj.right
        
    @property
    def simplify(self):
        return self.left.simplify + self.right.simplify

class OpGroup(Base):
    def __str__(self):
        mystr = ""
        for iarg,arg in enumerate(self.args):
            mystr += f"({arg})" if not(isinstance(arg,Symbol) or isinstance(arg,Unary)) else f"{arg}"
            if iarg < len(self.args) - 1:
                mystr += f"{self.operators[iarg]}"
        return mystr

    @property
    def simplify(self):
        return self.make_new_group(
            args=[arg.simplify for arg in self.args],
            operators=self.operators
        )

class AddSubGroup(OpGroup):
    def __init__(self, args, operators):
        self.args = args
        self.operators = operators

        print(f'ASgroup: args = {self.args}, ops = {self.operators}')

        assert len(self.args)-1 == len(self.operators)
        for op in operators:
            assert op in ["+", "-"]
    
    @property
    def class_order(self):
        """order of operations"""
        return 1
    
    @classmethod
    def cast(cls, obj, new_op=[]):
        if isinstance(obj, Add):
            return AddSubGroup(args=[obj.left, obj.right], operators=new_op + ["+"])
        elif isinstance(obj,Subtract):
            return AddSubGroup(args=[obj.left, obj.right], operators=new_op + ["-"])
        elif isinstance(obj,AddSubGroup):
            return obj
        else:
            return AddSubGroup(args=[obj], operators=new_op)
    
    @classmethod
    def make_new_group(cls, args, operators):
        """make a new group by calling __ op overloaded methods to get simplifications"""
        for iarg,arg in enumerate(args):
            if iarg == 0:
                res = arg
            else:
                op = operators[iarg-1]
                if op == "+":
                    res = res + arg
                else:
                    res = res - arg
        return res
    
    @property
    def _operators(self):
        return ["+"] + self.operators
    
    def derivative(self, sym, order=1):
        return self.make_new_group(
            args=[arg.derivative(sym,order) for arg in self.args],
            operators=self.operators
        )
    
    def __eq__(self, obj):
        if not isinstance(obj, AddSubGroup):
            return False
        elif len(obj.args) != len(self.args):
            return False
        else:
            return all([self.args[iarg] == obj.args[iarg] for iarg in range(len(self.args))]) and \
                all([self.operators[iop] == obj.operators[iop] for iop in range(len(self.operators))])
        
    @property
    def simplify(self):
        """reorder the products in class order"""
        operators = self._operators # includes first arg omitted operator * now so equal lengths
        dict_list = [{"arg" : self.args[i].simplify, "op" : operators[i]} for i in range(len(self.args))]
        sorted_dict_list = sorted(dict_list, key=lambda x : x["arg"].class_order + 10*(x["op"] == "-"))
        return AddSubGroup(
            args=[mdict["arg"] for mdict in sorted_dict_list],
            operators=[mdict["op"] for mdict in sorted_dict_list][1:],
        )   
    
    # TODO : add like-terms simplification
    
class MultDivGroup(OpGroup):
    def __init__(self, args, operators):
        self.args = args
        self.operators = operators
        print(f'MDgroup: args = {self.args}, ops = {self.operators}')

        assert len(self.args)-1 == len(self.operators)
        for op in operators:
            assert op in ["*", "/"]

    @property
    def _operators(self):
        return ["*"] + self.operators
        
    @property
    def class_order(self):
        """order of operations"""
        return 2
    
    @classmethod
    def cast(cls, obj, new_op=[]):
        if isinstance(obj, Multiply):
            return MultDivGroup(args=[obj.left, obj.right], operators=new_op + ["*"])
        elif isinstance(obj,Divide):
            return MultDivGroup(args=[obj.left, obj.right], operators=new_op + ["/"])
        elif isinstance(obj,MultDivGroup):
            return obj
        else:
            return MultDivGroup(args=[obj], operators=new_op)
    
    @classmethod
    def make_new_group(cls, args, operators):
        """make a new group by calling __ op overloaded methods to get simplifications"""
        for iarg,arg in enumerate(args):
            if iarg == 0:
                res = arg
            else:
                op = operators[iarg-1]
                if op == "*":
                    res = res * arg
                else:
                    res = res / arg
        return res

    def derivative(self, sym, order=1):
        # apply product/quotient rule here (diff one term at a time)
        for iarg,arg in enumerate(self.args):
            new_term = self / arg * arg.derivative(sym,order)
            if iarg == 0:
                res = new_term
            else:
                if self.operators[iarg-1] == "*":
                    res = res + new_term
                else:
                    res = res - new_term
        return res
    
    def __eq__(self, obj):
        if not isinstance(obj, MultDivGroup):
            return False
        elif len(obj.args) != len(self.args):
            return False
        else:
            return all([self.args[iarg] == obj.args[iarg] for iarg in range(len(self.args))]) and \
                all([self.operators[iop] == obj.operators[iop] for iop in range(len(self.operators))])
        
    @property
    def simplify(self):
        """reorder the products in class order"""
        operators = self._operators # includes first arg omitted operator * now so equal lengths
        dict_list = [{"arg" : self.args[i].simplify, "op" : operators[i]} for i in range(len(self.args))]
        sorted_dict_list = sorted(dict_list, key=lambda x : x["arg"].class_order + 10*(x["op"] == "/"))
        resorted_group =  MultDivGroup(
            args=[mdict["arg"] for mdict in sorted_dict_list],
            operators=[mdict["op"] for mdict in sorted_dict_list][1:],
        )      

        # # then combine like terms with floats and symbols out front
        # new_args = resorted_group.args
        # new_ops = resorted_group._operators # accounts for up front operator too
        # all_symbols = [_ for _ in range(len(self.args))]
        return resorted_group

    # TODO : add like-terms simplification 

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
    
    @property
    def class_order(self):
        """order of operations"""
        return 2
    
    def derivative(self, sym, order=1):
        # product rule
        left = self.left.derivative(sym,order) * self.right
        right = self.left * self.right.derivative(sym,order)
        return left + right
    
    def __eq__(self, obj):
        if not isinstance(obj, Multiply):
            return False
        # commutative equality
        elif self.left == obj.left and self.right == obj.right:
            return True
        elif self.left == obj.right and self.right == obj.left:
            return True
        else:
            return False
        
    @property
    def simplify(self):
        if self.left.class_order > self.right.class_order:
            return self.right.simplify * self.left.simplify
        else:
            return self.left.simplify * self.right.simplify
    
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
    
    @property
    def class_order(self):
        """order of operations"""
        return 2
    
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
        
    @property
    def simplify(self):
        return self.left.simplify / self.right.simplify
    
class Unary(Base):
    def __init__(self, arg):
        self.arg = arg

    @property
    def unary_name(self):
        return None
    
    @property
    def class_order(self):
        """order of operations"""
        return 3
    
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

    def __eq__(self, obj):
        return isinstance(obj,Sin) and self.arg == obj.arg  
    
    @property
    def simplify(self):
        return Sin(self.arg.simplify)

class Cos(Unary):
    def __init__(self, arg):
        self.arg = arg

    @property
    def unary_name(self):
        return "cos"
    
    def derivative(self, sym, order=1):
        # chain rule on sine
        return Sin(self.arg) * self.arg.derivative(sym,order) * Float(-1.0)
    
    def __eq__(self, obj):
        return isinstance(obj,Cos) and self.arg == obj.arg

    @property
    def simplify(self):
        return Cos(self.arg.simplify)  