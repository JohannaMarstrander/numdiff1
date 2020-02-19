import numpy as np
from Task1_a import BVP, solve_bvp, plott

def f(x, y):
    return 1 + 0 * x + 0 * y

def u(x, y):
    return 0 * x + 0 * y

def v(x, y):
    return np.array([y, -x])


M = 100
x, y = np.ogrid[0:1:(M + 1) * 1j, 0:1:(M + 1) * 1j]
ex = BVP(f, v, u, 0, 1, 0.01)
U = solve_bvp( BVP(f, v, u, 0, 1, 0.01), M)
plott(x, y, U.reshape((M + 1, M + 1)))
