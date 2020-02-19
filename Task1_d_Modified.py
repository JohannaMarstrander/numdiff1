import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from Task1_a import plott,BVP,I,rhs

def fdm_modified(bvp, M):
    """Creates the finite difference matrix with modified scheme with diriclet conditions"""
    A = np.zeros(((M + 1) ** 2, (M + 1) ** 2))
    h = (bvp.b - bvp.a) / M
    x, y = np.ogrid[0:1:(M + 1) * 1j, 0:1:(M + 1) * 1j]

    for i in range(1, M):
        for j in range(1, M):
            A[I(i, j, M), I(i, j, M)] = 4 * bvp.mu + bvp.v(x[j], x[i])[0] * h - bvp.v(x[j], x[i])[1] * h
            A[I(i, j, M), I(i - 1, j, M)] = -bvp.mu  # - bvp.v(x[j], x[i])[1] * h / 2
            A[I(i, j, M), I(i + 1, j, M)] = -bvp.mu + bvp.v(x[j], x[i])[1] * h
            A[I(i, j, M), I(i, j - 1, M)] = -bvp.mu - bvp.v(x[j], x[i])[0] * h
            A[I(i, j, M), I(i, j + 1, M)] = -bvp.mu  # + bvp.v(x[j], x[i])[0] * h

    # Incorporate boundary conditions
    # Add boundary values related to unknowns from the first and last grid ROW
    for j in [0, M]:
        for i in range(0, M + 1):
            A[I(i, j, M), I(i, j, M)] = h ** 2

    # Add boundary values related to unknowns from the first and last grid COLUMN
    for i in [0, M]:
        for j in range(1, M + 1):
            A[I(i, j, M), I(i, j, M)] = h ** 2

    return sparse.csr_matrix(A)


# Function for solving the bvp
def solve_bvp_modified(bvp, M):
    A = fdm_modified(bvp, M)
    F = rhs(bvp, M)
    U = spsolve(A, F)
    return U


def f(x, y):
    return 1 + 0 * x + 0 * y  #


def u(x, y):
    return 0 * x + 0 * y


def v(x, y):
    return np.array([y, -x])


M = 10
x, y = np.ogrid[0:1:(M + 1) * 1j, 0:1:(M + 1) * 1j]
U = solve_bvp_modified(BVP(f, v, u, 0, 1, 0.01), M)
plott(x, y, U.reshape((M + 1, M + 1)))
