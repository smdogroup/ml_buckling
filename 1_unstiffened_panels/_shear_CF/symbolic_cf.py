import ml_buckling as mlb

# not the prettiest looking code for symbolic manipulation here
# but the point is functionality over prettiness
# I'm coding a symbolic manipulation specific to this problem
# to hopefully be able to integrate the shear-closed form solution
# using expanded sinusoidal terms (all expanded to one factor only) Sean Engelstad
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
print(mysine)