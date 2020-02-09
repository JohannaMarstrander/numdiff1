import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from Task1 import BVP,solve_bvp,plott




# Make a test problem:
def f(x, y):
    return (4*x + 2*y - 0.06)
def u_ex(x, y):
    return (2*x**2 + y**2 )
def v(x,y):
    return np.array([1,1])

ex_1 = BVP(f, v, u_ex, 0, 1, 0.01, u_ex)

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


U_ny=np.reshape(U,(M+1,M+1))

plott(x,y,U_ny)
