import ml_buckling as mlb
import unittest


class TestSingleOp(unittest.TestCase):
    def test_constructor(self):
        print("test constructor")
        print(mlb.Symbol("a").name)
        assert mlb.Symbol("a").name == "a"

    def test_add1(self):
        print("test add1")
        expr = mlb.Symbol("a", float=2.0) + mlb.Symbol("a", float=1.14)
        print(expr)
        assert expr == mlb.Symbol("a", float=3.14)

    def test_add2(self):
        print("test add2")
        print(mlb.Symbol("a", float=2.0) + mlb.Symbol("b", float=1.14))

    def test_sub1(self):
        print("test sub1")
        expr = mlb.Symbol("a", float=2.0) - mlb.Symbol("a", float=1.14)
        print(expr)
        assert expr == mlb.Symbol("a", float=0.86)

    def test_sub2(self):
        print("test sub2")
        print(mlb.Symbol("a", float=2.0) - mlb.Symbol("b", float=1.14))

    def test_multiply1(self):
        print("test multiply1")
        expr = mlb.Symbol("a", float=2.0) * mlb.Symbol("a")
        print(expr)
        assert expr == mlb.Symbol("a", float=2.0, exponent=2.0)

    def test_multiply2(self):
        print("test multiply2")
        print(mlb.Symbol("a", float=2.0) * mlb.Symbol("b", float=1.0))

    def test_div1(self):
        print("test div1")
        expr = mlb.Symbol("a", float=2.0) / mlb.Symbol("a")
        print(expr)
        assert expr == mlb.Float(2.0)

    def test_div2(self):
        print("test div2")
        print(mlb.Symbol("a", float=2.0) / mlb.Symbol("b", float=1.0))


if __name__ == "__main__":
    unittest.main()
    # tester = TestSingleOp()
    # tester.test_div2()
