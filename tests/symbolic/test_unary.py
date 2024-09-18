import ml_buckling as mlb
import unittest


class TestUnary(unittest.TestCase):
    def test_sin(self):
        print("test sin()")
        x1 = mlb.Symbol("x1")
        x2 = mlb.Symbol("x2")
        expr = mlb.Sin(x1 + x2)
        print(expr)

    def test_sin2(self):
        print("test sin()")
        x1 = mlb.Symbol("x1")
        x2 = mlb.Symbol("x2")
        term1 = mlb.Sin(x1 + x2)
        term2 = mlb.Sin(x1 - x2)
        print(term1 + term2)


if __name__ == "__main__":
    unittest.main()
