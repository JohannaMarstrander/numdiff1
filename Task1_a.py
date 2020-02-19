import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy import sparse
from scipy.sparse.linalg import spsolve


def I(i, j, n):
    """Define index mapping"""
    return i + j * (n + 1)


class BVP(object):
    """ Define problem class for general BVP given in task 1  """

    def __init__(self, f, v, g=0, a=0, b=1, mu=1, uexact=None):
        self.f = f  # Source function
        self.g = g  # Function for boundary condition, valid on the boundary
        self.a = a  # Interval
        self.b = b
        self.mu = mu
        self.v = v
        self.uexact = uexact  # The exact solution, if known.


def fdm(bvp, M):
    """Creates the finite difference matrix with diriclet conditions"""
    A = np.zeros(((M + 1) ** 2, (M + 1) ** 2))
    h = (bvp.b - bvp.a) / M
    x, y = np.ogrid[0:1:(M + 1) * 1j, 0:1:(M + 1) * 1j]

    for i in range(1, M): #incorporate the difference scheme on the inner gridpoints
        for j in range(1, M):
            A[I(i, j, M), I(i, j, M)] = 4 * bvp.mu  # P
            A[I(i, j, M), I(i - 1, j, M)] = (-bvp.mu - bvp.v(x[j], x[i])[1] * h / 2)
            A[I(i, j, M), I(i + 1, j, M)] = (-bvp.mu + bvp.v(x[j], x[i])[1] * h / 2)
            A[I(i, j, M), I(i, j - 1, M)] = (-bvp.mu - bvp.v(x[j], x[i])[0] * h / 2)
            A[I(i, j, M), I(i, j + 1, M)] = (-bvp.mu + bvp.v(x[j], x[i])[0] * h / 2)

    # Incorporate boundary conditions
    # Add boundary values related to unknowns from the first and last grid ROW
    for j in [0, M]:
        for i in range(0, M + 1):
            A[I(i, j, M), I(i, j, M)] = h ** 2

    # Add boundary values related to unknowns from the first and last grid COLUMN
    for i in [0, M]:
        for j in range(1, M + 1):
            A[I(i, j, M), I(i, j, M)] = h ** 2

    return sparse.csr_matrix(A)  # Transform to sparse matrix for faster calculations


def rhs(bvp, M):
    """Function for creating rhs of eq depending on f and g"""
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


def solve_bvp(bvp, M):
    """Function for solving the bvp"""
    A = fdm(bvp, M)
    F = rhs(bvp, M)
    U = spsolve(A, F)

    return U


def plott(x, y, Z):
    """Function for 3D plotting,
    edited from https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(x, y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.view_init(30, 110)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
