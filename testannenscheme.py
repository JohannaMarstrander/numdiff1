import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from Task1 import plott

# PROBLEM 1)
# a) Solve the problem on the unit square with Dirichlet conditions
# Set up and implement a finite differences scheme using central differences
# Write the problem so it can solve the problem with different values for v, f and BCs.

# Define index mapping
def I(i, j, n):
    return i + j * (n + 1)


# Define problem class for general BVP given in task 1
class BVP(object):
    def __init__(self, f, v, g=0, a=0, b=1, mu=1, uexact=None):
        self.f = f  # Source function
        self.g = g  # Function for boundary condition, valid on the boundary
        self.a = a  # Interval
        self.b = b
        self.mu = mu
        self.v = v
        self.uexact = uexact  # The exact solution, if known.



def fdm(bvp, M):
    A = np.zeros(((M + 1) ** 2, (M + 1) ** 2))
    h = (bvp.b - bvp.a) / M
    x, y = np.ogrid[0:1:(M + 1) * 1j, 0:1:(M + 1) * 1j]

    for i in range(1, M):
        for j in range(1, M):
            A[I(i, j, M), I(i, j, M)] = 4 * bvp.mu  + bvp.v(x[j], x[i])[0] * h - bvp.v(x[j], x[i])[1] * h
            A[I(i, j, M), I(i - 1, j, M)] = -bvp.mu #- bvp.v(x[j], x[i])[1] * h / 2
            A[I(i, j, M), I(i + 1, j, M)] = -bvp.mu + bvp.v(x[j], x[i])[1] * h
            A[I(i, j, M), I(i, j - 1, M)] = -bvp.mu - bvp.v(x[j], x[i])[0] * h
            A[I(i, j, M), I(i, j + 1, M)] = -bvp.mu #+ bvp.v(x[j], x[i])[0] * h

    # Incorporate boundary conditions
    # Add boundary values related to unknowns from the first and last grid ROW
    for j in [0, M]:
        for i in range(0, M + 1):
            A[I(i, j, M), I(i, j, M)] = h ** 2

    # Add boundary values related to unknowns from the first and last grid COLUMN
    for i in [0, M]:
        for j in range(1, M + 1):
            A[I(i, j, M), I(i, j, M)] = h ** 2

    # for i in range(0,(M+1)**2):
    # print(A[i,:])
    return A


# Function for creating rhs of eq depending on f and g
def rhs(bvp, M):
    x, y = np.ogrid[0:1:(M + 1) * 1j, 0:1:(M + 1) * 1j]
    h = (bvp.b - bvp.a) / M

    F = (bvp.f(x, y)).ravel()
    G = (bvp.g(x, y)).ravel()

    # Add boundary values related to unknowns from the first and last grid ROW
    bc_indices = [I(i, j, M) for j in [0, M] for i in range(0, M + 1)]

    F[bc_indices] = G[bc_indices]

    # Add boundary values related to unknowns from the first and last grid COLUMN
    bc_indices = [I(i, j, M) for i in [0, M] for j in range(0, M + 1)]
    F[bc_indices] = G[bc_indices]

    return F * h ** 2


# Function for solving the bvp
def solve_bvp(bvp, M):
    A = fdm(bvp, M)
    F = rhs(bvp, M)
    U = la.solve(A, F)
    return U


def plott(x, y, Z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(x, y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

# def f(x, y):
#    return (v(x, y)[0] * 4 * x + v(x, y)[1] * 2 * y - 6)
#
#
# def u(x, y):
#    return (2 * x ** 2 + y ** 2)
#
#
# def v(x, y):
#    return np.array([ 1, 1])
#
#
#
# M = 40
# ex_1 = BVP(f, v, u, 0, 1, 1,u)
#
# x,y = np.ogrid[0:1:(M+1)*1j, 0:1:(M+1)*1j]
#
#
# U = solve_bvp(ex_1, M)
#
# plott(x,y,U.reshape((M + 1, M + 1)))
# plott(x,y,u(x,y))
#
# U_ext = ex_1.uexact(x, y).ravel()
#
# error = U_ext - U
# Eh = np.linalg.norm(error, ord=np.inf)
# print('The error is {:.2e}'.format(Eh))


def f1(x, y):
    return 1 + 0 * x + 0 * y  # Ettersom python noen ganger er totalt retard


def u(x, y):
    return 0 * x + 0 * y


def v(x, y):
    return np.array([y, -x])


M_list = [10]
for M1 in M_list:
    x, y = np.ogrid[0:1:(M1 + 1) * 1j, 0:1:(M1 + 1) * 1j]
    ex = BVP(f1, v, u, 0, 1, 0.01)
    U = solve_bvp(ex, M1)
    U_ny = U.reshape((M1 + 1, M1 + 1))
    plott(x, y, U_ny)