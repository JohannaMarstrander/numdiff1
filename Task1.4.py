import numpy as np
from Task1 import BVP, solve_bvp, plott


# def u(x,y):
#     return np.sin(np.pi*2*x) + np.cos(np.pi*2*y)
# def f1(x,y):
#     return (0.01*((2*np.pi)**2*np.sin(np.pi*2*x) + (2*np.pi)**2 * np.cos(np.pi*2*y))
#            +y*(2*np.pi)*np.cos(np.pi*2*x) + x*(2*np.pi)*np.sin(np.pi*2*y))
# def v(x,y):
#     return np.array([y,-x])

def f1(x, y):
    return 1 + 0 * x + 0 * y  # Ettersom python noen ganger er totalt retard


def u(x, y):
    return 0 * x + 0 * y


def v(x, y):
    return np.array([y, -x])





M_list = [20]
# M_list = [20]
E = []
h_list = []
U_list = []
for M1 in M_list:
    x, y = np.ogrid[0:1:(M1 + 1) * 1j, 0:1:(M1 + 1) * 1j]
    ex = BVP(f1, v, u, 0, 1, 0.01)
    U = solve_bvp(ex, M1)
    U_ny = U.reshape((M1 + 1, M1 + 1))
    plott(x, y, U_ny)
    U_list.append(U)

