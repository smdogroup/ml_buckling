import ml_buckling as mlb
import unittest

class TestCompund(unittest.TestCase):
    def test_one(self):
        print("test compound one")
        expr = mlb.Symbol("a", float=2.145) * mlb.Symbol("b", float=13.05, exponent=2)
        expr += mlb.Symbol("c", exponent=0.5)
        print(expr)

    def test_two(self):
        print("test compound two")
        expr = mlb.Symbol("a", float=2.145) / mlb.Symbol("b", float=13.05, exponent=2)
        expr += mlb.Symbol("c", exponent=0.5)
        expr *= mlb.Symbol("d") + mlb.Symbol("Ergfalo")
        print(expr)

if __name__=="__main__":
    unittest.main()