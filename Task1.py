import numpy as np
import scipy.linalg as la


# PROBLEM 1)
# a) Solve the problem on the unit square with Dirichlet conditions
# Set up and implement a finite differences scheme using central differences
# Write the problem so it can solve the problem with different values for v, f and BCs.

# Define index mapping
def I(i, j, n):
    return i + j * (n + 1)


# Define problem class for general BVP given in task 1
class BVP(object):
    def __init__(self, f, v, g=0, a=0, b=1, mu= 1, uexact=None):
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
    A = np.zeros(((M+1)**2,(M+1)**2))
    h = (bvp.b - bvp.a)/M
    
    for i in range(1,M):  
        for j in range(1, M): 
            A[I(i,j,M),I(i,j,M)] = 4 * bvp.mu # P
            A[I(i,j,M),I(i-1,j,M)] = -bvp.mu - bvp.v(i-1,j)[0] *  h/2 # W
            A[I(i,j,M),I(i+1,j,M)] = -bvp.mu + bvp.v(i+1,j)[0] * h/2 # E
            A[I(i,j,M),I(i,j-1,M)] = -bvp.mu - bvp.v(i,j-1)[1] * h/2 # S
            A[I(i,j,M),I(i,j+1,M)] = -bvp.mu + bvp.v(i,j+1)[1] * h/2 # N
    
    # Incorporate boundary conditions
    # Add boundary values related to unknowns from the first and last grid ROW
    for j in [0,M]:
        for i in range(0,M+1):
            A[I(i,j,M),I(i,j,M)] = h**2

    # Add boundary values related to unknowns from the first and last grid COLUMN
    for i in [0,M]:
        for j in range(1,M+1):
           A[I(i,j,M),I(i,j,M)] = h**2
            
    return A

# Function for creating rhs of eq depending on f and g
def rhs(bvp, M):
    x,y = np.ogrid[0:1:(M+1)*1j, 0:1:(M+1)*1j]
    h = (bvp.b - bvp.a)/M
    
    F = (bvp.f(x,y)).ravel()
    G = (bvp.g(x,y)).ravel()
    
     # Add boundary values related to unknowns from the first and last grid ROW
    bc_indices = [ I(i,j,M)  for j in [0, M] for i in range(0, M+1) ]
    F[bc_indices] = G[bc_indices]  

    # Add boundary values related to unknowns from the first and last grid COLUMN
    bc_indices = [ I(i,j,M) for i in [0, M] for j in range(0, M+1)]
    F[bc_indices] = G[bc_indices]
    
    return F*h**2

# Function for solving the bvp
def solve_bvp(bvp, M):
    A = fdm(bvp, M)
    F = rhs(bvp,M)
    U = la.solve(A, F)
    return U
    

# Make a test problem: 
def f(x, y):
    return (4*x + 2*y - 6)
def u_ex(x, y):
    return (2*x**2 + y**2 )
def v(x,y):
    return np.array([1,1])

ex_1 = BVP(f, v, u_ex, 0, 1, 1, u_ex) 

# Number of subdivisions in each dimension
M = 5

#Define the grid using a sparse grid, and using the imaginary number 1j to include the endpoints
x,y = np.ogrid[0:1:(M+1)*1j, 0:1:(M+1)*1j]

# Evaluate u on the grid.
U_ext= ex_1.uexact(x, y).ravel()
U = solve_bvp(ex_1, M)

print("Numerical sol:", U)
print("Exact sol:" ,U_ext)

error = U_ext-U
print(error)
Eh = np.linalg.norm(error,ord=np.inf) 
print('The error is {:.2e}'.format(Eh))

    
