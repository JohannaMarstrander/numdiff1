import numpy as np
from Task1 import I, plott, BVP
from scipy import sparse
from scipy.sparse.linalg import spsolve


def fdm_neumann(bvp, M):
    A = np.zeros(((M + 1) ** 2, (M + 1) ** 2))
    h = (bvp.b - bvp.a) / M
    x, y = np.ogrid[0:1:(M + 1) * 1j, 0:1:(M + 1) * 1j]

    for i in range(1, M):
        for j in range(1, M):
            A[I(i, j, M), I(i, j, M)] = 4 * bvp.mu  # P
            A[I(i, j, M), I(i - 1, j, M)] = (-bvp.mu - bvp.v(x[j], x[i])[1] * h / 2)  # W
            A[I(i, j, M), I(i + 1, j, M)] = (-bvp.mu + bvp.v(x[j], x[i])[1] * h / 2)  # E
            A[I(i, j, M), I(i, j - 1, M)] = (-bvp.mu - bvp.v(x[j], x[i])[0] * h / 2)  # S
            A[I(i, j, M), I(i, j + 1, M)] = (-bvp.mu + bvp.v(x[j], x[i])[0] * h / 2)  # N

    # Incorporate boundary conditions
    # Add boundary values related to unknowns from the first and last grid ROW
    for j in [0, M]:
        for i in range(1, M + 1):
            A[I(i, j, M), I(i, j, M)] = h ** 2

    # Add boundary values related to unknowns from the first and last grid COLUMN
    for j in range(0, M + 1):
        A[I(M, j, M), I(M, j, M)] = h ** 2
    # for i in [M]:
    #    for j in range(0, M + 1):
    #        A[I(i, j, M), I(i, j, M)] = h ** 2

    # Neumannconditions
    # for j in [M]:
    #    for i in range(1, M ): #tror dette gir mening for å ikke få med hjørner
    #        A[I(i, j, M), I(i, j, M)] = 4 * bvp.mu
    #        A[I(i, j, M), I(i, j-1, M)] = -2*bvp.mu
    #        A[I(i, j, M), I(i-1, j , M)] = -bvp.mu - bvp.v(x[j], x[i])[0] * h / 2
    #        A[I(i, j, M), I(i + 1, j, M)] = -bvp.mu + bvp.v(x[j], x[i])[0] * h / 2

    for i in [0]:
        for j in range(0, M):
            A[I(i, j, M), I(i, j, M)] = 4 * bvp.mu
            A[I(i, j, M), I(i + 1, j, M)] = -2 * bvp.mu
            A[I(i, j, M), I(i, j + 1, M)] = -bvp.mu + bvp.v(x[j], x[i])[0]
            A[I(i, j, M), I(i, j - 1, M)] = -bvp.mu - bvp.v(x[j], x[i])[0]

    A[I(0, M, M), I(0, M, M)] = 4 * bvp.mu
    A[I(0, M, M), I(0, M - 1, M)] = -2 * bvp.mu
    A[I(0, M, M), I(0 + 1, M, M)] = -2 * bvp.mu

    # siste hjørnet
    # A[I(0,0,M),I(0,0,M)] = 4 * bvp.mu
    # A[I(0, 0, M), I(0, 1, M)] = -2 * bvp.mu
    # A[I(0, 0, M), I(1, 0, M)] = -2 * bvp.mu

    return sparse.csr_matrix(A)


# Function for creating rhs of eq depending on f and g
def rhs_neumann(bvp, M):
    x, y = np.ogrid[0:1:(M + 1) * 1j, 0:1:(M + 1) * 1j]
    h = (bvp.b - bvp.a) / M

    F = (bvp.f(x, y)).ravel()
    G = (bvp.g(x, y)).ravel()

    # Add boundary values related to unknowns from the first and last grid ROW
    bc_indices = [I(i, j, M) for j in [0, M] for i in range(1, M)]
    F[bc_indices] = G[bc_indices]

    # Add boundary values related to unknowns from the first and last grid COLUMN
    bc_indices = [I(M, j, M) for j in range(0, M + 1)]
    F[bc_indices] = G[bc_indices]

    return F * h ** 2


# Function for solving the bvp
def solve_bvp(bvp, M):
    A = fdm_neumann(bvp, M)
    F = rhs_neumann(bvp, M)
    U = spsolve(A, F)
    return U


def f1(x, y):
    return 1 + 0 * x + 0 * y  # Ettersom python noen ganger er totalt retard


def u(x, y):
    return 0 * x + 0 * y


def v(x, y):
    return np.array([y, -x])


M_list = [20]
for M1 in M_list:
    x, y = np.ogrid[0:1:(M1 + 1) * 1j, 0:1:(M1 + 1) * 1j]
    ex = BVP(f1, v, u, 0, 1, 0.01)
    U = solve_bvp(ex, M1)
    U_ny = U.reshape((M1 + 1, M1 + 1))
    plott(x, y, U_ny)
