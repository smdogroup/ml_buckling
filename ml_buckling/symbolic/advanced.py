__all__ = ["SymbolGroup", "DualSin", "DualCos", "Fraction"]

from .symbol import *
import numpy as np

# class AdditionGroup:
#     def __init__(self)

class Fraction:
    @classmethod
    def zero(cls):
        return cls(num=0.0)
    
    @classmethod
    def one(cls):
        return cls(num=1.0)
    
    @classmethod
    def mone(cls):
        return cls(num=-1.0)
    
    @classmethod
    def cast(cls, float):
        if isinstance(float, Fraction):
            return float
        elif np.mod(float,1) == 0.0: # it is an integer
            return cls(num=float)
        else:
            raise AssertionError("Can't cast this float to fraction")

    def __init__(self, num, den=1):
        self.num = num
        self.den = den

    def __str__(self):
        num = int(self.num)
        den = int(self.den)
        if self.den == 1.0:
            return f"{num}"
        else:
            return f"{num}/{den}"
    
    def __eq__(self, obj):
        if isinstance(obj,float):
            return self.num == obj * self.den
        elif isinstance(obj,Fraction):
            return self.num == obj.num and self.den == obj.den
        
    @property
    def simplify(self):
        """simplify denominator by dividing out lcm"""
        gcd = np.gcd(int(self.num), int(self.den))
        return Fraction(
            num=int(self.num/gcd),
            den=int(self.den/gcd)
        )

    def __add__(self, obj):
        if isinstance(obj,float):
            return self + self.cast(obj)
        elif isinstance(obj,int):
            return self + self.cast(obj)
        elif isinstance(obj,Fraction):
            # a/b + c/d = (ad + bc) / bd
            return Fraction(
                num=self.num * obj.den + self.den * obj.num,
                den=self.den * obj.den
            ).simplify
        
    def __mul__(self, obj):
        if isinstance(obj,float):
            return self * self.cast(obj)
        elif isinstance(obj,int):
            return self * self.cast(obj)
        elif isinstance(obj,Fraction):
            # a/b * c/d = ac/bd
            return Fraction(
                num=self.num * obj.num,
                den=self.den * obj.den
            ).simplify
        elif isinstance(obj,SymbolGroup):
            return obj * self # refer to symbol group
        
    def __truediv__(self, obj):
        if isinstance(obj,Fraction):
            # a/b / (c/d) = ad/bc
            return Fraction(
                num=self.num * obj.den,
                den=self.den * obj.num
            ).simplify
    

class SymbolGroup:
    @classmethod
    def from_letter(cls, letter, float=1.0):
        return SymbolGroup.from_symbols(
            [Symbol(letter, float=float)]
        )

    @classmethod
    def from_symbols(cls, symbols):
        mfloat = np.prod(np.array([symbol.float for symbol in symbols]))
        return SymbolGroup(
            float=Fraction.cast(mfloat),
            letters=[symbol.name for symbol in symbols],
            exponents=[symbol.exponent for symbol in symbols],
        )

    def __init__(self, float, letters, exponents):
        self.float = Fraction.cast(float)
        self.float = float
        self.letters = letters
        self.exponents = exponents

    @property
    def simplify(self):
        # get unique list of letters
        unique_letters = list(np.unique(self.letters))
        unique_exponents = []
        for letter in unique_letters:
            exponent = 0.0
            for j,_letter in enumerate(self.letters):
                if letter == _letter:
                    exponent += self.exponents[j]
            unique_exponents += [exponent]
        return SymbolGroup(
            float=self.float,
            letters=unique_letters,
            exponents=unique_exponents,
        )
    
    def __add__(self, obj):
        if isinstance(obj,SymbolGroup):
            return AddGroup([self,obj])
        elif isinstance(obj,AddGroup):
            return AddGroup([self] + obj.objs)

    def __mul__(self, obj):
        if isinstance(obj, SymbolGroup):
            return SymbolGroup(
                float=self.float * obj.float,
                letters=self.letters + obj.letters,
                exponents=self.exponents + obj.exponents
            ).simplify
        elif isinstance(obj,Fraction):
            return SymbolGroup(
                float=self.float * obj,
                letters=self.letters,
                exponents=self.exponents
            ).simplify
        elif isinstance(obj, DualSin):
            return obj * self
        elif isinstance(obj, DualCos):
            return obj * self
        elif isinstance(obj, AddGroup):
            return obj * self # flip order to prevent redundancy
        
    def __truediv__(self, obj):
        if isinstance(obj, SymbolGroup):
            return SymbolGroup(
                float=self.float / obj.float,
                letters=self.letters + obj.letters,
                exponents=self.exponents + [-exp for exp in obj.exponents]
            ).simplify
    
    def __str__(self):
        case = 2
        if case == 1: # regular style
            if self.float == 0.0:
                return ""
            mstr = f"{self.float}" if self.float != 1.0 else ""
            if len(self.letters) > 0 and self.float != 1.0:
                mstr += "*"
            for i,letter in enumerate(self.letters):
                mstr += f"{letter}^{self.exponents[i]}" if self.exponents[i] != 1.0 else f"{letter}"
                if i < len(self.letters)-1:
                    mstr += "*"
            return mstr
        elif case == 2: # somewhat pretty print style
            if self.float == 0.0:
                return ""
            mstr = f"{self.float}"

            for i,letter in enumerate(self.letters):
                if self.exponents[i] > 0:
                    mstr += "*"
                elif self.exponents[i] < 0:
                    mstr += "/"

                if self.exponents[i] == 0.0:
                    mstr += ""
                elif self.exponents[i] in [1,-1]:
                    mstr += f"{letter}"
                else:
                    mstr += f"{letter}^{abs(int(self.exponents[i]))}"
            return mstr
    
    def set_float(self, new_float):
        # maybe use this to set float with method cascading
        self.float = new_float
        return self
    
    @classmethod
    def zero(cls):
        return cls.num(0.0)

    @classmethod
    def one(cls):
        return cls.num(1.0)
    
    @classmethod
    def num(cls, num):
        num = Fraction.cast(num)
        return cls(float=num, letters=[], exponents=[])
    
    @classmethod
    def Pi(cls):
        return cls(float=1, letters=["Pi"], exponents=[1])
    
    def derivative(self, letter, order=1):
        # sparse derivatives since only one letter can match (this is good)
        # no N terms or something
        # first simplify this object
        simp = self.simplify
        exponent = None
        for i,_letter in enumerate(simp.letters):
            if letter == _letter:
                exponent = simp.exponents[i]
                break
        if exponent is None:
            return SymbolGroup(
                float=0.0,
                letters=[],
                exponents=[],
            )
        else:
            new_exponents = simp.exponents
            new_exponents[i] -= 1
            first_deriv = SymbolGroup(
                float=simp.float * exponent,
                letters=simp.letters,
                exponents=new_exponents
            )
            if order > 1:
                return first_deriv.derivative(letter,order)
            else:
                return first_deriv
            
    def antiderivative(self, letter, order=1):
        simp = self.simplify
        exponent = None
        for i,_letter in enumerate(simp.letters):
            if letter == _letter:
                exponent = simp.exponents[i]
                break
        if exponent is None:
            return simp * SymbolGroup.from_letter(letter)
        else:
            new_exponents = simp.exponents
            new_exponents[i] += 1
            first_integral = SymbolGroup(
                float=simp.float / (exponent+1),
                letters=simp.letters,
                exponents=new_exponents
            )
            if order > 1:
                return first_integral.antiderivative(letter,order)
            else:
                return first_integral
    
class DualSin:
    def __init__(self, arg1, arg2, const=None, coeff=None):
        self.arg1 = arg1 #x1 coeff
        self.arg2 = arg2 #x2 coeff
        if const is None:
            const = SymbolGroup.zero()
        self.const = const
        if coeff is None:
            coeff = SymbolGroup.one()
        # coeff is a symbolGroup
        self.coeff = coeff

    @property
    def float(self):
        return self.coeff.float

    # def __add__(self, obj):
    #     return None

    @classmethod
    def copy(cls, obj):
        return cls(
            arg1=obj.arg1,
            arg2=obj.arg2,
            const=obj.const,
            coeff=obj.coeff
        )

    def __mul__(self, obj):
        if isinstance(obj,SymbolGroup):
            new_self = self.copy(self)
            new_self.coeff = self.coeff * obj
            return new_self
        # trig product to sum identities
        elif isinstance(obj,DualCos):
            # TODO : need ways to simplify the internal addGroups stuff in the
            # args and consts
            left = DualSin(
                arg1=AddGroup([self.arg1,obj.arg1]),
                arg2=AddGroup([self.arg2, obj.arg2]),
                const=AddGroup([self.const, obj.const]),
                coeff=SymbolGroup.num(Fraction(1,2))*self.coeff*obj.coeff,
            )
            right = DualSin(
                arg1=AddGroup([self.arg1,obj.arg1 * SymbolGroup.num(-1.0)]),
                arg2=AddGroup([self.arg2, obj.arg2 * SymbolGroup.num(-1.0)]),
                const=AddGroup([self.const, obj.const * SymbolGroup.num(-1.0)]),
                coeff=SymbolGroup.num(Fraction(1,2))*self.coeff*obj.coeff,
            )
            return AddGroup([left, right])
        
        elif isinstance(obj,DualSin):
            # TODO : need ways to simplify the internal addGroups stuff in the
            # args and consts
            left = DualCos(
                arg1=AddGroup([self.arg1,obj.arg1 * SymbolGroup.num(-1.0)]),
                arg2=AddGroup([self.arg2, obj.arg2 * SymbolGroup.num(-1.0)]),
                const=AddGroup([self.const, obj.const * SymbolGroup.num(-1.0)]),
                coeff=SymbolGroup.num(Fraction(1,2))*self.coeff*obj.coeff,
            )
            right = DualCos(
                arg1=AddGroup([self.arg1,obj.arg1]),
                arg2=AddGroup([self.arg2, obj.arg2]),
                const=AddGroup([self.const, obj.const]),
                coeff=SymbolGroup.num(Fraction(-1,2))*self.coeff*obj.coeff,
            )
            return AddGroup([left, right])
        
        # distributive property (with AddGroup)
        elif isinstance(obj,AddGroup):
            return AddGroup([
                self*_obj*obj.factor for _obj in obj.objs
            ])
        
    def derivative(self, sym, order=1):
        """derivative with respect to a particular symbol"""
        assert sym in ["x1", "x2"] # for now as I don't want to multiply by add groups yet
        assert isinstance(order, int) and order >= 1
        if sym == "x1":
            first_deriv = DualCos(
                arg1=self.arg1,
                arg2=self.arg2,
                const=self.const,
                coeff=self.coeff * self.arg1,
            )
        elif sym == "x2":
            first_deriv = DualCos(
                arg1=self.arg1,
                arg2=self.arg2,
                const=self.const,
                coeff=self.coeff * self.arg2,
            )
        if order == 1:
            return first_deriv
        else:
            # recursive call for higher order derivatives
            return first_deriv.derivative(sym,order-1)
        
    def integral(self, symbol, lower, upper):
        # this is a definite integral of the sine expression
        pass
        
    @property
    def simplify(self):
        return DualSin(
            arg1=self.arg1.simplify,
            arg2=self.arg2.simplify,
            const=self.const.simplify,
            coeff=self.coeff.simplify
        )

    def __str__(self):
        mystr = str(self.coeff)
        if isinstance(self.coeff, AddGroup):
            mystr = f"({mystr})"
        if mystr != "":
            mystr += "*"
        mystr += f"Sin([{self.arg1}]*x1+[{self.arg2}]*x2+[{self.const}])"
        return mystr
    
    # TODO : need some kind of pretty print feature that uses
    # a common denominator property
    
class DualCos(DualSin):
    def __init__(self, arg1, arg2, const=None, coeff=None):
        super(DualCos,self).__init__(arg1, arg2, const, coeff)

    def __mul__(self, obj):
        if isinstance(obj,SymbolGroup):
            new_self = self.copy(self)
            new_self.coeff = self.coeff * obj
            return new_self
        elif isinstance(obj, DualSin):
            return obj * self # switch order to prevent duplicate code
        elif isinstance(obj,DualCos):
            # TODO : need ways to simplify the internal addGroups stuff in the
            # args and consts
            left = DualCos(
                arg1=AddGroup([self.arg1,obj.arg1 * SymbolGroup.num(-1.0)]),
                arg2=AddGroup([self.arg2, obj.arg2 * SymbolGroup.num(-1.0)]),
                const=AddGroup([self.const, obj.const * SymbolGroup.num(-1.0)]),
                coeff=SymbolGroup.num(Fraction(1,2))*self.coeff*obj.coeff,
            )
            right = DualCos(
                arg1=AddGroup([self.arg1,obj.arg1]),
                arg2=AddGroup([self.arg2, obj.arg2]),
                const=AddGroup([self.const, obj.const]),
                coeff=SymbolGroup.num(Fraction(1,2))*self.coeff*obj.coeff,
            )
            return AddGroup([left, right])
        # distributive property (with AddGroup)
        elif isinstance(obj,AddGroup):
            return AddGroup([
                self*_obj*obj.factor for _obj in obj.objs
            ])

    def derivative(self, sym, order=1):
        """derivative with respect to a particular symbol"""
        assert sym in ["x1", "x2"] # for now as I don't want to multiply by add groups yet
        assert isinstance(order, int) and order >= 1
        if sym == "x1":
            first_deriv = DualSin(
                arg1=self.arg1,
                arg2=self.arg2,
                const=self.const,
                coeff=self.coeff * self.arg1 * SymbolGroup.num(-1.0),
            )
        elif sym == "x2":
            first_deriv = DualSin(
                arg1=self.arg1,
                arg2=self.arg2,
                const=self.const,
                coeff=self.coeff * self.arg2 * SymbolGroup.num(-1.0),
            )
        if order == 1:
            return first_deriv
        else:
            # recursive call for higher order derivatives
            return first_deriv.derivative(sym,order-1)

    @property
    def simplify(self):
        return DualCos(
            arg1=self.arg1.simplify,
            arg2=self.arg2.simplify,
            const=self.const.simplify,
            coeff=self.coeff.simplify
        )

    def __str__(self):
        mystr = str(self.coeff)
        if isinstance(self.coeff, AddGroup):
            mystr = f"({mystr})"
        if mystr != "":
            mystr += "*"
        mystr += f"Cos([{self.arg1}]*x1+[{self.arg2}]*x2+[{self.const}])"
        return mystr
    
class AddGroup:
    """encompasses additiona and subtraction together (with negation for subtraction instead)"""
    def __init__(self, objs, factor=None):
        self.objs = objs
        if factor is None:
            factor = SymbolGroup.num(1)
        self.factor = factor

    @property
    def float(self):
        return None # does not have defined float

    @property
    def _distr_factor(self):
        return AddGroup([obj*self.factor for obj in self.objs], None)

    @property
    def _remove_nesting(self):
        _nested_add = any([isinstance(obj,AddGroup) for obj in self.objs])
        if _nested_add:
            mlist = []
            for obj in self.objs:
                if isinstance(obj,AddGroup):
                    new_objs = [_obj * obj.factor for _obj in obj.objs]
                    mlist += new_objs
                else:
                    mlist += [obj]
            new_group = AddGroup(mlist, self.factor)
            return new_group._distr_factor
        else:
            return self

    @property
    def simplify(self):
        # convert back to regular no factor
        new_group = self._remove_nesting

        # prune any terms in the add group that have float == 0.0
        new_group = AddGroup([obj.simplify for obj in new_group.objs if not(obj.float == 0.0)],
                             new_group.factor.simplify)
        new_group = new_group._distr_factor
        
        # reduce nested AddGroups
        new_group = new_group._remove_nesting

        # apply common factor to add group
        common_factor = new_group.common_factor
        # only works on all SymbolicGroups
        if common_factor is not None and not(common_factor == SymbolGroup.num(1.0)):
            new_group = AddGroup(objs=[
                obj/common_factor for obj in new_group.objs
            ], factor=common_factor*new_group.factor)
        
        # check if there is only one entry then remove nesting
        if len(new_group.objs) == 1:
            new_group = new_group.objs[0] * new_group.factor
        elif len(new_group.objs) == 0:
            new_group = SymbolGroup.zero()

        if isinstance(new_group,AddGroup):
            new_group = new_group._combine_like_terms()

        # final return from simplify
        return new_group
    
    def derivative(self, sym, order=1):
        """derivative with respect to a particular symbol"""
        first_deriv = AddGroup([
               obj.derivative(sym,order) for obj in self.objs
            ], factor=self.factor) + AddGroup(
                self.objs, self.factor.derivative(sym,order)
        )
        if order == 1:
            return first_deriv
        else:
            # recursive call for higher order derivatives
            return first_deriv.derivative(sym,order-1)
    
    # TODO : need to add distribute/expand method
    # TODO : need to add common denominator method (maybe have optional denominator in arg..)
    
    def _combine_like_terms(self):
        """combine like terms for simplification"""
        all_symbolic = all([isinstance(obj,SymbolGroup) for obj in self.objs])
        if not all_symbolic:
            return self
        else: # all symbolic
            # make a dictionary of all the exponents in each symbol
            all_letters = []
            for obj in self.objs:
                all_letters += obj.letters
            all_letters = list(np.unique(all_letters))

            exp_dicts = []
            for obj in self.objs:
                exp_dict = {}
                for letter in all_letters:
                    _found = False
                    for i,_letter in enumerate(obj.letters):
                        if letter == _letter:
                            exp_dict[letter] = obj.exponents[i]
                            _found = True; break
                    if not _found:
                        exp_dict[letter] = 0

                exp_dicts += [exp_dict]

            # now check dict pairs for equality
            # building up a new list
            new_list = []
            for i,obj in enumerate(self.objs):
                exp_dict = exp_dicts[i]
                _match = False
                for j,obj2 in enumerate(new_list):
                    exp_dict2 = exp_dicts[j]
                    if exp_dict == exp_dict2:
                        # then combine these objects
                        new_list[j].float = new_list[j].float + obj.float
                        _match = True; break
                if not _match:
                    new_list += [obj]

            # now return the modified AddGroup list
            return AddGroup(new_list, self.factor)

    def __add__(self, obj):
        if isinstance(obj, SymbolGroup):
            return obj + self # reduce code duplication
        elif isinstance(obj,AddGroup):
            # need to distribute factors into the terms first
            objs1 = [obj1*self.factor for obj1 in self.objs]
            objs2 = [obj2*obj.factor for obj2 in obj.objs]

            return AddGroup(objs1 + objs2)

    def __mul__(self, obj):
        if isinstance(obj, DualSin):
            return obj * self # reverse order to prevent duplicated code
        elif isinstance(obj, AddGroup):
            # distributive property on two AddGroups
            new_list = []
            for obj1 in self.objs:
                for obj2 in obj.objs:
                    new_list += [obj1 * obj2]
            return AddGroup(new_list, self.factor * obj.factor)
        elif isinstance(obj, SymbolGroup):
            return AddGroup(self.objs,factor=self.factor*obj)
        
    @property
    def common_factor(self) -> SymbolGroup:
        # require that all objs in the AddGroup are SymbolicGroups
        all_symbolic = all([isinstance(obj,SymbolGroup) for obj in self.objs])
        if not all_symbolic:
            return None
        else: # all symbolic
            all_letters = []
            lcm = 1
            for obj in self.objs:
                all_letters += obj.letters
                lcm = np.lcm(lcm, obj.float.den)
            den_float = SymbolGroup.num(1) / SymbolGroup.num(lcm)
            all_letters = list(np.unique(all_letters))
            max_neg_exps = [100 for _ in range(len(all_letters))]
            for obj in self.objs:
                _found = [False for _ in all_letters]
                for i,letter1 in enumerate(obj.letters):
                    for j,letter2 in enumerate(all_letters):
                        if letter1 == letter2:
                            exp = obj.exponents[i]
                            _found[j] = True
                            if exp < max_neg_exps[j]:
                                max_neg_exps[j] = exp
                                break

                for j in range(len(all_letters)):
                    if not _found[j]:
                        exp = 0
                        if exp < max_neg_exps[j]:
                            max_neg_exps[j] = exp

            return SymbolGroup(
                float=den_float.float,
                letters=all_letters,
                exponents=max_neg_exps,
            )
                    


    def __str__(self):
        mystr = ""
        if self.factor is not None:
            mystr += str(self.factor) + "*["
        for i,obj in enumerate(self.objs):
            mystr += f"({obj})"
            if i < len(self.objs) - 1:
                mystr += "+"
        if self.factor is not None:
            mystr += "]"
        return mystr