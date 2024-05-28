import ml_buckling as mlb
import unittest

class TestAdvanced(unittest.TestCase):
    def test_advanced1(self):
        print("test advanced1")
        group = mlb.SymbolGroup.from_symbols([
            mlb.Symbol("a", float=2.0), mlb.Symbol("b", exponent=2.0), mlb.Symbol("c", exponent=-1.0),
            mlb.Symbol("a", exponent=3.0, float=1.15),
        ])
        print(group)
        print("simplify: " + str(group.simplify))

    def test_deriv1(self):
        print("test deriv1")
        group = mlb.SymbolGroup.from_symbols([
            mlb.Symbol("a", float=2.0), mlb.Symbol("b", exponent=2.0), mlb.Symbol("c", exponent=-1.0),
            mlb.Symbol("a", exponent=3.0, float=3.0),
        ])
        print(f"deriv of group {group.simplify} w.r.t. a order 1:")
        print(group.derivative("a", order=1))
        print(f"then also d/dc")
        print(group.derivative("a", order=1).derivative("c", order=1))

    def test_mode_shape1(self):
        print("\ntest mode shape1")
        mysine = mlb.DualSin(
            arg1=mlb.SymbolGroup.from_symbols([
                mlb.Symbol("Pi"), mlb.Symbol("lam1", exponent=-1)
            ]), # x1 coeff
            arg2=mlb.SymbolGroup.from_symbols([
                mlb.Symbol("Pi", float=-1.0), mlb.Symbol("lam2"),
                mlb.Symbol("lam1", exponent=-1)
            ]), # x2 coeff
            const=mlb.SymbolGroup.from_symbols([
                mlb.Symbol("Pi", float=mlb.Fraction(-1,2)),
            ]), # const term
            coeff=None # if None defaults to 1 SymbolGroup
        )
        mygroup = mlb.SymbolGroup.from_letter("a")
        expr = mygroup * mysine
        print(expr)
        expr2 = mysine * mygroup
        print(expr2)

    def test_trig_product(self):
        print("\ntest trig_product")
        mysine = mlb.DualSin(
            arg1=mlb.SymbolGroup.from_symbols([
                mlb.Symbol("Pi"), mlb.Symbol("lam1", exponent=-1)
            ]), # x1 coeff
            arg2=mlb.SymbolGroup.from_symbols([
                mlb.Symbol("Pi", float=-1.0), mlb.Symbol("lam2"),
                mlb.Symbol("lam1", exponent=-1)
            ]), # x2 coeff
            const=mlb.SymbolGroup.from_symbols([
                mlb.Symbol("Pi", float=-0.5),
            ]), # const term
            coeff=None # if None defaults to 1 SymbolGroup
        )

        mycos = mlb.DualCos(
            arg1=mlb.SymbolGroup.from_symbols([
                mlb.Symbol("Pi")
            ]), # x1 coeff
            arg2=mlb.SymbolGroup.zero(), # x2 coeff
            const=mlb.SymbolGroup.from_symbols([
                mlb.Symbol("Pi", float=-0.5),
            ]), # const term
            coeff=None # if None defaults to 1 SymbolGroup
        )

        # need to add some way to get a common denominator
        # and have AddGroup in the numerator with simplification of terms
        # maybe make this a method in the AddGroup

        expr = mysine * mycos
        expr = expr.simplify
        print(expr)

    def test_sine_deriv(self):
        print("\ntest sine deriv")
        mysine = mlb.DualSin(
            arg1=mlb.SymbolGroup.from_symbols([
                mlb.Symbol("Pi"), mlb.Symbol("lam1", exponent=-1)
            ]), # x1 coeff
            arg2=mlb.SymbolGroup.from_symbols([
                mlb.Symbol("Pi", float=-1.0), mlb.Symbol("lam2"),
                mlb.Symbol("lam1", exponent=-1)
            ]), # x2 coeff
            const=mlb.SymbolGroup.from_symbols([
                mlb.Symbol("Pi", float=-0.5),
            ]), # const term
            coeff=None # if None defaults to 1 SymbolGroup
        )
        print(f"sine = {mysine}")
        print(f"deriv = {mysine.derivative("x1").simplify}")

    def test_sine_integral(self):
        print("\ntest sine integral")
        # sin(pi*x1) integrated
        mysine = mlb.DualSin(
            arg1=mlb.SymbolGroup.from_symbols([
                mlb.Symbol("Pi")
            ]), # x1 coeff
            arg2=mlb.SymbolGroup.zero()
        )
        # want to eval integral at two periodic locations

        # integral = mysine.
        print(f"sine = {mysine}")
        print(f"deriv = {mysine.derivative("x1").simplify}")

    def test_mode_shape3(self):
        # test multiplying the full mode shape together for w
        # and then differentiating twice
        print("\ntest modeshape3")

        # again not the prettiest code (but we want functionality here)

        # factor 1 = 0.5 * np.sin(pi * (X1 - lam2 * X2) *m/a - pi/2) where m is positive integer
        mysine = mlb.DualSin(
            arg1=mlb.SymbolGroup.from_symbols([
                mlb.Symbol("Pi"), mlb.Symbol("a", exponent=-1), mlb.Symbol("m")
            ]), # x1 coeff
            arg2=mlb.SymbolGroup.from_symbols([
                mlb.Symbol("Pi", float=-1), mlb.Symbol("lam2"), 
                mlb.Symbol("a", exponent=-1), mlb.Symbol("m")
            ]), # x2 coeff
            const=mlb.SymbolGroup.from_symbols([
                mlb.Symbol("Pi", float=mlb.Fraction(-1,2)),
            ]), # const term
            coeff=None # if None defaults to 1 SymbolGroup
        )     
        print(f"step 1 : " + str(mysine)) 

        # factor 2 = np.sin(np.pi * X1 / a)
        expr = mysine * mlb.DualSin(
            arg1=mlb.SymbolGroup.from_symbols([
                mlb.Symbol("Pi"), mlb.Symbol("a", exponent=-1)
            ]),
            arg2=mlb.SymbolGroup.zero(),
        )
        expr = expr.simplify  
        print(f"step 2 : " + str(expr)) 

        # factor 2 = np.sin(np.pi * X1 / a)
        w = expr * mlb.DualSin(
            arg1=mlb.SymbolGroup.zero(),
            arg2=mlb.SymbolGroup.from_symbols([
                mlb.Symbol("Pi"), mlb.Symbol("b", exponent=-1)
            ]),
        )
        print(f"step 3 : " + str(w)) 
        w = w.simplify # this is the main mode shape now
        print(f"\nstep 4 : " + str(w)) 

        # now differentiate it twice
        print("\nnow differentiate twice::")
        w1 = w.derivative("x1")
        w1 = w1.simplify
        print(f"\nd/dx1(w) = {w1}")
        
        w11 = w1.derivative("x1").simplify
        print(f"\nd^2/dx1^2(w) = {w11}")

    def test_fraction(self):
        frac = mlb.Fraction(1,2) + mlb.Fraction(3,5)
        print(frac)

    def test_common_factor(self):
        print(f"\ntest common factor")
        term1 = mlb.SymbolGroup.from_symbols([
            mlb.Symbol("Pi"), mlb.Symbol("lam1", exponent=-1)
        ])
        term2 = mlb.SymbolGroup.from_symbols([
            mlb.Symbol("Pi", float=-1.0), mlb.Symbol("lam2"),
            mlb.Symbol("lam1", exponent=-1)
        ]) * mlb.Fraction(1,5)
        term3 = mlb.SymbolGroup.from_symbols([
            mlb.Symbol("Pi"), mlb.Symbol("b", exponent=-1)
        ])
        expr = term1 + term2 + term3
        print(f"orig expr = {expr}")
        print(f"common factor = {expr.common_factor}")

        expr = expr.simplify
        print(f"factored expression = {expr}")

        # apply common factor in simplification

    def test_deriv_add_group(self):
        print(f"\ntest deriv add group")
        term1 = mlb.SymbolGroup.from_symbols([
            mlb.Symbol("Pi"), mlb.Symbol("lam1", exponent=-1)
        ])
        term2 = mlb.SymbolGroup.from_symbols([
            mlb.Symbol("Pi", float=-1.0), mlb.Symbol("lam2"),
            mlb.Symbol("lam1", exponent=-1)
        ]) * mlb.Fraction(1,5)
        term3 = mlb.SymbolGroup.from_symbols([
            mlb.Symbol("Pi"), mlb.Symbol("b", exponent=-1)
        ])
        expr = term1 + term2 + term3
        print(f"orig expr = {expr}")
        print(f"common factor = {expr.common_factor}")

        expr = expr.simplify
        print(f"factored expression = {expr}")

        deriv = expr.derivative("lam1", order=1)
        print(f"deriv = {deriv}")
        print(f"simplified = {deriv.simplify}")

    def test_orig_mode_shape(self):
        # test multiplying the full mode shape together for w
        # and then differentiating twice
        print("\ntest orig mode shape")

        # again not the prettiest code (but we want functionality here)

        # factor 1 = 0.5 * np.sin(np.pi * (X1 - lam2 * X2 - lam1 / 2.0) / lam1)
        mysine = mlb.DualSin(
            arg1=mlb.SymbolGroup.from_symbols([
                mlb.Symbol("Pi"), mlb.Symbol("lam1", exponent=-1)
            ]), # x1 coeff
            arg2=mlb.SymbolGroup.from_symbols([
                mlb.Symbol("Pi", float=-1), mlb.Symbol("lam2"),
                mlb.Symbol("lam1", exponent=-1)
            ]), # x2 coeff
            const=mlb.SymbolGroup.from_symbols([
                mlb.Symbol("Pi", float=mlb.Fraction(-1,2)),
            ]), # const term
            coeff=None # if None defaults to 1 SymbolGroup
        )     
        print(f"step 1 : " + str(mysine)) 

        # factor 2 = np.sin(np.pi * X1 / a)
        w = mysine * mlb.DualSin(
            arg1=mlb.SymbolGroup.zero(),
            arg2=mlb.SymbolGroup.from_symbols([
                mlb.Symbol("Pi"), mlb.Symbol("b", exponent=-1)
            ]),
        )
        print(f"step 3 : " + str(w)) 
        w = w.simplify # this is the main mode shape now
        print(f"\nstep 4 : " + str(w)) 

        # now differentiate it twice
        print("\nnow differentiate twice::")
        w1 = w.derivative("x1")
        w1 = w1.simplify
        print(f"\nd/dx1(w) = {w1}")
        
        w11 = w1.derivative("x1").simplify
        print(f"\nd^2/dx1^2(w) = {w11}")


if __name__=="__main__":
    # unittest.main()
    tester = TestAdvanced()
    tester.test_mode_shape3()
    # tester.test_orig_mode_shape()
    # tester.test_common_factor()
    # tester.test_deriv_add_group()