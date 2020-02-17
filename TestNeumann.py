import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


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


# A function for creating the finite difference matrix
# NOTE: Dirichlet conditions

def fdm(bvp, M):
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
    for j in [0,M]:
        for i in range(1, M + 1):
            A[I(i, j, M), I(i, j, M)] = h ** 2

    # Add boundary values related to unknowns from the first and last grid COLUMN
    for i in [M]:
        for j in range(0, M + 1):
            A[I(i, j, M), I(i, j, M)] = h ** 2


    #Neumannconditions
    #ToDO: Sjekk alle range
    #for j in [M]:
    #    for i in range(1, M ): #tror dette gir mening for å ikke få med hjørner
    #        A[I(i, j, M), I(i, j, M)] = 4 * bvp.mu
    #        A[I(i, j, M), I(i, j-1, M)] = -2*bvp.mu
    #        A[I(i, j, M), I(i-1, j , M)] = -bvp.mu - bvp.v(x[j], x[i])[0] * h / 2
    #        A[I(i, j, M), I(i + 1, j, M)] = -bvp.mu + bvp.v(x[j], x[i])[0] * h / 2

    for i in [0]:
        for j in range(0, M ):
            A[I(i, j, M), I(i, j, M)] = 4 * bvp.mu
            A[I(i, j, M), I(i+1, j, M)] = -2*bvp.mu
            A[I(i, j, M), I(i, j+1 , M)] = -bvp.mu + bvp.v(x[j], x[i])[0]
            A[I(i, j, M), I(i , j-1, M)] = -bvp.mu - bvp.v(x[j], x[i])[0]

    A[I(0,M,M),I(0,M,M)]= 4 * bvp.mu
    A[I(0, M, M), I(0, M-1, M)] = -2 * bvp.mu
    A[I(0, M, M), I(0+1, M, M)] = -2 * bvp.mu

    #siste hjørnet
    #A[I(0,0,M),I(0,0,M)] = 4 * bvp.mu
    #A[I(0, 0, M), I(0, 1, M)] = -2 * bvp.mu
    #A[I(0, 0, M), I(1, 0, M)] = -2 * bvp.mu


    return A


# Function for creating rhs of eq depending on f and g
def rhs(bvp, M):
    x, y = np.ogrid[0:1:(M + 1) * 1j, 0:1:(M + 1) * 1j]
    h = (bvp.b - bvp.a) / M

    F = (bvp.f(x, y)).ravel()
    G = (bvp.g(x, y)).ravel()

    # Add boundary values related to unknowns from the first and last grid ROW
    bc_indices = [I(i, j, M) for j in [0,M] for i in range(1, M )]
    F[bc_indices] = G[bc_indices]

    # Add boundary values related to unknowns from the first and last grid COLUMN
    bc_indices = [I(i, j, M) for i in [M] for j in range(0, M + 1)]
    F[bc_indices] = G[bc_indices]

    #T = np.zeros((M+1)*(M+1))
    #bc_indices = [I(i, j, M) for j in [M] for i in range(0, M + 1)]
    #F[bc_indices] = T[bc_indices]

    #bc_indices = [I(i, j, M) for i in [0] for j in range(0, M + 1)]
    #F[bc_indices] = T[bc_indices]

    #y=0 og x=1, i=0,J=M
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
    #ax.set_zlim(-1, 2)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def f1(x, y):
    return 1 + 0*x + 0*y #Ettersom python noen ganger er totalt retard
def u(x, y):
    return 0*x + 0*y
def v(x,y):
    return np.array([y,-x])


#ex_1 = BVP(f1, v, u, 0, 1, 0.01)
#Define the grid using a sparse grid, and using the imaginary number 1j to include the endpoints
#x,y = np.ogrid[0:1:(M+1)*1j, 0:1:(M+1)*1j]

# Evaluate u on the grid.
#U_ext= ex_1.uexact(x, y).ravel()
#U = solve_bvp(ex_1, M)



M_list=[20,30,40]
#M_list = [20]
E=[]
h_list=[]
U_list=[]
for M1 in M_list:
    x, y = np.ogrid[0:1:(M1 + 1) * 1j, 0:1:(M1 + 1) * 1j]
    ex = BVP(f1, v, u, 0, 1, 0.01)
    U = solve_bvp(ex, M1)
    U_ny=U.reshape((M1+1,M1+1))
    #U_ny = np.reshape(U, (M1 + 1, M1 + 1))
    #print(U_ny)
    plott(x,y,U_ny)
    U_list.append(U)

