import numpy as np, matplotlib.pyplot as plt
from matplotlib import cm
import argparse
import os, math

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--case", type=int, default=8)
parent_parser.add_argument("--show", type=bool, default=False)
parent_parser.add_argument("--AR", type=float, default=2.0)
parent_parser.add_argument("--lam2", type=float, default=1.0)

args = parent_parser.parse_args()

# do simple case of square plate first
# try to get mode shape to match the plate BCs if possible
b = 1.0
a = b * args.AR

m1 = 1  # one orthogonal mode

A = 1.0
m1 = math.ceil(args.AR)
lam1 = 2.0 * a / m1
print(f"lam1 = {lam1}")
# lam1 = 2.0
lam2 = args.lam2  # probably not actually 1.0 (eqn for this)

# make 3d plot
nx1 = 50
nx2 = 50
x1vec = np.linspace(0, a, nx1)
x2vec = np.linspace(0, b, nx2)
X1, X2 = np.meshgrid(x1vec, x2vec)
Wflat = 0.0 * X1

if args.case == 1:
    # just diagonal in one direction
    W = A * np.sin(np.pi * (X1 + lam2 * X2) / lam1)

elif args.case == 2:
    # diagonal in two directions, but narrower the other way
    W = A * np.sin(np.pi * (X1 + lam2 * X2) / lam1)
    # narrower width in other direction
    lam1_2 = 1.4
    W *= np.cos(np.pi * (X2 - lam2 * X1) / lam1_2)
    W *= 2.0

elif args.case == 3:
    # diagonal in two directions, but narrower the other way
    W = A * np.sin(np.pi * (X1 + lam2 * X2) / lam1)
    # narrower width in other direction
    lam1_2 = 1.4
    W *= np.cos(np.pi * (X2 - lam2 * X1) / lam1_2)

    # now multiply by regular sinusoids in each direction
    W *= np.sin(np.pi * X1 / a)
    W *= np.sin(np.pi * X2 / b)
    # ends up being too close to the axial mode shape as these regular sinusoids modify it too much

elif args.case == 4:
    # now instead of case 3 multiply by regular sinusoids => mutiply by superellipse to modify the original mode shape less
    # diagonal in two directions, but narrower the other way
    A = 0.5
    W = A * np.sin(np.pi * (X1 + lam2 * X2) / lam1)
    # narrower width in other direction
    lam1_2 = 1.6
    W *= np.cos(np.pi * (X2 - lam2 * X1) / lam1_2)

    # now multiply by superellipses in each direction to make the BCs at plate simply supported
    nx = 4.0
    ny = 4.0
    W *= np.power(np.sin(np.pi * X1 / a), 2.0 / nx)
    W *= np.power(np.sin(np.pi * X2 / b), 2.0 / ny)

elif args.case == 5:
    # now instead of case 3 multiply by regular sinusoids => mutiply by superellipse to modify the original mode shape less
    # diagonal in two directions, but narrower the other way
    A = 0.5
    W = A * np.sin(np.pi * (X1 + lam2 * X2) / lam1)
    # narrower width in other direction
    # except now make diagonal direction like a clamped in this direction
    # lam1_2 = 1.6
    clamped_func = 0.5 * (1.0 + np.cos(2.0 * np.pi * (X2 - lam2 * X1) / lam1))
    W *= clamped_func

    # now multiply by superellipses in each direction to make the BCs at plate simply supported
    nx = 4.0
    ny = 4.0
    W *= np.power(np.sin(np.pi * X1 / a), 2.0 / nx)
    W *= np.power(np.sin(np.pi * X2 / b), 2.0 / ny)

elif args.case == 6:
    # now instead of case 3 multiply by regular sinusoids => mutiply by superellipse to modify the original mode shape less
    # diagonal in two directions, but narrower the other way
    A = 0.5
    W = A * np.sin(np.pi * (X1 + lam2 * X2) / lam1)
    # narrower width in other direction
    # except now make diagonal direction like a clamped in this direction
    # lam1_2 = 1.6
    clamped_func = 0.5 * (1.0 + np.cos(2.0 * np.pi * (X2 - lam2 * X1) / lam1))
    W *= clamped_func ** 2.0

    # now multiply by superellipses in each direction to make the BCs at plate simply supported
    nx = 4.0
    ny = 4.0
    W *= np.power(np.sin(np.pi * X1 / a), 2.0 / nx)
    W *= np.power(np.sin(np.pi * X2 / b), 2.0 / ny)

elif args.case == 7:
    m1 = math.ceil(args.AR)  # really ceil(rho0)
    print(f"m1 = {m1}")
    # technically min m1 slightly differs
    lam1 = a / m1

    W = 0.5 * np.sin(np.pi * (X1 - lam2 * X2 - lam1 / 2.0) / lam1)
    W *= np.sin(np.pi * X1 / lam1) ** 2
    W *= (
        np.sin(np.pi * X2 / b) ** 2
    )  # X2**2 might be optional TBD on that (but might make it more accurate with 2 half-waves in 2-direction for instance subcase)

elif args.case == 8:
    m1 = math.ceil(args.AR)  # really ceil(rho0)
    print(f"m1 = {m1}")
    # technically min m1 slightly differs
    lam1 = a / m1

    W = 0.5 * np.sin(np.pi * (X1 - lam2 * X2 - lam1 / 2.0) / lam1)
    # W *= np.sin(np.pi * X1 / a)**2
    W *= np.sin(np.pi * X1 / a)
    W *= np.sin(np.pi * X2 / b)
    # TODO : for cases 7 and 8 the sin(x) thing seems to restrictive that it won't work with other lam2 values
    # probably need to compare directly against the mode shape.. eigenvector
    # this is the lowest order that is slightly closer to getting true feasible mode shapes..


# TODO : add case 7 here for debugging and trying new things with AR = 2.0 case..
# compare to the actual modes..

# now multiply by a super ellipse
# n = 4.0 # higher order makes it more box-like
# W *= np.power(np.sin(np.pi * X1/a), 2.0/n)
# W *= np.power(np.sin(np.pi * X2/b), 2.0/n)

# multiply by superellipse in x2 direction alsoz

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, zorder=False)
ax.plot_surface(
    X1,
    X2,
    Wflat,
    cmap=cm.coolwarm,
    linewidth=0,
    alpha=0.5,
    antialiased=False,
    zorder=0.5,
)

ax.plot_surface(X1, X2, W, cmap=cm.coolwarm, linewidth=0, antialiased=False, zorder=2)

# plot one point to make equal aspect ratio
ax.scatter(a, a, a)

ax.set_zlim(0.0, a)
plt.xlabel("X1")
plt.ylabel("X2")
if args.show:
    plt.show()
ax.view_init(elev=40.0, azim=-90 - 10)

# make a folder for the current AR
AR_folder = f"AR_{args.AR}"
if not os.path.exists(AR_folder):
    os.mkdir(AR_folder)

plt.savefig(f"{AR_folder}/case{args.case}.png", dpi=400)
