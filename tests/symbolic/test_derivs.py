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



    # def test_deriv2(self):

if __name__=="__main__":
    unittest.main()