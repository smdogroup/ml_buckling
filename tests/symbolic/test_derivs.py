import ml_buckling as mlb
import unittest

class TestDerivative(unittest.TestCase):
    def test_deriv1(self):
        expr = mlb.Symbol("a", float=2.0, exponent=3.0)
        deriv1 = expr.derivative("a", 1)
        deriv2 = expr.derivative("a", 2)
        print(f"d({expr})/da = {deriv1}")
        print(f"d^2({expr})/da^2 = {deriv2}")
        assert deriv1 == mlb.Symbol("a", float=6.0, exponent=2.0)
        assert deriv2 == mlb.Symbol("a", float=12.0, exponent=1.0)

    def test_deriv2(self):
        expr = mlb.Symbol("a", exponent=2.0) + mlb.Symbol("b", exponent=3.0)
        deriv = expr.derivative("a", 1)
        expr2 = mlb.Symbol("a", float=2.0, exponent=1.0)
        print(f"deriv {deriv} == {expr2}")
        assert deriv == expr2

    def test_sine(self):
        expr = mlb.Sin(mlb.Symbol("a") * mlb.Float(3.0))
        deriv = expr.derivative("a")
        print(f"d({expr})/da = {deriv}")
        expr2 = mlb.Float(3.0) * mlb.Cos(mlb.Float(3.0) * mlb.Symbol("a"))
        print(f"expr2 = {expr2}")
        assert deriv == expr2

    def test_sine2(self):
        print("\ntest sine 2")
        expr = mlb.Sin(mlb.Symbol("a") * mlb.Float(3.0)) + mlb.Cos(mlb.Symbol("b", float=2.0) + mlb.Symbol("a", float=-1.0)) * mlb.Symbol("a", exponent=2.0)
        deriv = expr.derivative("a").simplify
        print(f"d({expr})/da = {deriv}")
        # expr2 = mlb.Float(3.0) * mlb.Cos(mlb.Float(3.0) * mlb.Symbol("a"))
        # print(f"expr2 = {expr2}")
        # assert deriv == expr2

if __name__=="__main__":
    unittest.main()
    # tester = TestDerivative()
    # tester.test_sine()