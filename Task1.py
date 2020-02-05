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
    hh = h*h
    for i in range(1,M):  
        for j in range(1, M): 
            A[I(i,j,M),I(i,j,M)] = 4 * bvp.mu * 1/hh # P
            A[I(i,j,M),I(i-1,j,M)] = -bvp.mu/hh - bvp.v(i-1,j)[0] /(2*h) # W
            A[I(i,j,M),I(i+1,j,M)] = -bvp.mu/hh + bvp.v(i+1,j)[0] /(2*h) # E
            A[I(i,j,M),I(i,j-1,M)] = -bvp.mu/hh - bvp.v(i,j-1)[1] /(2*h) # S
            A[I(i,j,M),I(i,j+1,M)] = -bvp.mu/hh + bvp.v(i,j+1)[1] /(2*h) # N
    
    # Incorporate boundary conditions
    # Add boundary values related to unknowns from the first and last grid ROW
    for j in [0,M]:
        for i in range(0,M+1):
            A[I(i,j,M),I(i,j,M)] = 1

    # Add boundary values related to unknowns from the first and last grid COLUMN
    for i in [0,M]:
        for j in range(1,M+1):
           A[I(i,j,M),I(i,j,M)] = 1
            
    return A

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

# Evaluate u on the grid. The output will be a 2-dimensional array 
# where U_ex_grid[i,j] = u_ex(x_i, y_j)
U_ex= ex_1.uexact(x, y).ravel()
F = (ex_1.f(x,y)).ravel()

# Overskriver F
for j in [0,M]:
    for i in range(0,M+1):  
        F[I(i,j,M)] = ex_1.g(x[j],x[i])

for i in [0,M]:
    for j in range(1,M+1):
       F[I(i,j,M)] = ex_1.g(x[j],x[i])

A = fdm(ex_1, M)
print(A@U_ex)
print(F)
U = la.solve(A, F)
print("Numerical sol:", U)
print("Exact sol:" ,U_ex)

error = U_ex-U
print(error)
Eh = np.linalg.norm(error,ord=np.inf) 
print('The error is {:.2e}'.format(Eh))

    
