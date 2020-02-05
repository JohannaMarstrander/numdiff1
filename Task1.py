# PROBLEM 1)
# a) Solve the problem on the unit square with Dirichlet conditions
# Set up and implement a finite differences scheme using central differences
# Write the problem so it can solve the problem with different values for v, f and BCs.

# Define index mapping
def I(i, j, n):
    return i + j * (n + 1)


# Define problem class
class BVP(object):
    def __init__(self, f, ga=0, gb=0, a=0, b=1, uexact=None):
        self.f = f  # Source function
        self.ga = ga  # Left boundary condition
        self.gb = gb  # Right boundary condition
        self.a = a  # Interval
        self.b = b
        self.uexact = uexact  # The exact solution, if known.


# Auxillary function for creating a tridiagonal matrix A=tridiag(v, d, w) of dimension N x N.
def tridiag(v, d, w, N):
    e = np.ones(N)  # array [1,1,...,1] of length N
    A = v * np.diag(e[1:], -1) + d * np.diag(e) + w * np.diag(e[1:], 1)
    return A


# A function for creating the finite difference matrix
def fdm(bvp):
