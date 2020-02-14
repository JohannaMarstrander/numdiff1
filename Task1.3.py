# -*- coding: utf-8 -*-
from Task1 import I, BVP, plott
import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Make a test problem:
def f(x, y):
    return (v(x,y)[0]*(y-1) + v(x,y)[1]*(x-1))
def u_ex(x, y):
        return ((1-x)*(1-y))
def v(x,y):
    return np.array([x*y,x**2])

def f2(x, y):
    return (4 + v(x,y)[0]*(-2*x) + v(x,y)[1]*(-2*y))
def u_ex2(x, y):
        return (1-x**2 - y**2)


def shape_matrix(M):
    A = np.ones((M+1,M+1))
    for i in range(M+1):
        for j in range(M+1):
            if ((i/M)**2 + (j/M)**2)>1: 
                A[i,j] = 0
    for i in range(M):
      for j in range(M):
           if (A[i,j] == 1 and (A[i,j+1] == 0 or A[i+1, j] == 0)):
                A[i,j] = 2
    return A

def fdm_circle(bvp, M, S):
    A = np.zeros(((M+1)**2,(M+1)**2))
    h = (bvp.b - bvp.a)/M
    x,y = np.ogrid[0:1:(M+1)*1j, 0:1:(M+1)*1j]

    #Nanna tror i går langs y-aksen, og j-langs x aksen og dermed er retningene motsatt av det som står under
    for i in range(1,M):  
        for j in range(1, M): 
            if S[i,j] == 0: #Utenfor området. 
                A[I(i,j,M), I(i,j,M)] = h**2 
            elif S[i,j] == 1: #Punkter der alt er OK: både E, W, S, N er definert. 
                A[I(i,j,M),I(i,j,M)] = 4 * bvp.mu # P
                A[I(i,j,M),I(i-1,j,M)] = (-bvp.mu - bvp.v(x[j],x[i])[1] *  h/2) # W
                A[I(i,j,M),I(i+1,j,M)] = (-bvp.mu + bvp.v(x[j],x[i])[1] * h/2) # E
                A[I(i,j,M),I(i,j-1,M)] = (-bvp.mu - bvp.v(x[j],x[i])[0] * h/2) # S
                A[I(i,j,M),I(i,j+1,M)] = (-bvp.mu + bvp.v(x[j],x[i])[0] * h/2) # N
            elif S[i, j] == 2:
                theta_x = (np.sqrt(M*M - i*i)-j)
                theta_y = (np.sqrt(M*M - j*j)-i)
                
                if theta_y < 1 and theta_y > 0:           
                    A[I(i,j,M),I(i,j,M)] += 2 * bvp.mu / theta_y
                    A[I(i,j,M),I(i-1,j,M)] = -2*bvp.mu/(1+theta_y)  - bvp.v(x[j],x[i])[1] *  h/(2*(1 + theta_y))
                else:
                    A[I(i,j,M),I(i,j,M)] += 2 * bvp.mu 
                    A[I(i,j,M),I(i-1,j,M)] = (-bvp.mu - bvp.v(x[j],x[i])[1] *  h/2) # W
                    A[I(i,j,M),I(i+1,j,M)] = (-bvp.mu + bvp.v(x[j],x[i])[1] * h/2) # E
                    
                if theta_x < 1 and theta_x > 0:   
                    A[I(i,j,M),I(i,j,M)] += 2 * bvp.mu / theta_x
                    A[I(i,j,M),I(i,j-1,M)] =  -2*bvp.mu/(1+theta_x)  - bvp.v(x[j],x[i])[0] *  h/(2*(1 + theta_x)) # S
                else: 
                    A[I(i,j,M),I(i,j,M)] += 2 * bvp.mu 
                    A[I(i,j,M),I(i,j-1,M)] = (-bvp.mu - bvp.v(x[j],x[i])[0] * h/2) # S
                    A[I(i,j,M),I(i,j+1,M)] = (-bvp.mu + bvp.v(x[j],x[i])[0] * h/2) # N
                  
    
    # Incorporate boundary conditions
    # Add boundary values related to unknowns from the first and last grid ROW
    for j in [0,M]:
        for i in range(0,M+1):
            A[I(i,j,M),I(i,j,M)] = h**2

    # Add boundary values related to unknowns from the first and last grid COLUMN
    for i in [0,M]:
        for j in range(1,M+1):
           A[I(i,j,M),I(i,j,M)] = h**2

    return sparse.csr_matrix(A)   # Transform to sparse matrix for faster calculations

def rhs_circle(bvp, M, S):
    x,y = np.ogrid[0:1:(M+1)*1j, 0:1:(M+1)*1j]
    h = (bvp.b - bvp.a)/M

    F = np.zeros((M+1)*(M+1)) 
    
    for i in range(M+1):
        for j in range(M+1):
            if (i == 0 or j == 0) and S[i,j]!=0:  
                F[I(i,j,M)] = bvp.g(x[j], y[:,i])
            elif S[i,j] != 0:
                F[I(i,j,M)] = bvp.f(x[j], y[:,i])
            if S[i,j] == 2: 
                theta_x = (np.sqrt(M*M - i*i)-j)
                theta_y = (np.sqrt(M*M - j*j)-i)
                if theta_x < 1 and theta_x > 0: 
                    F[I(i,j,M)] += 2 * bvp.mu / (h**2*(1+ theta_x)*theta_x) * bvp.g(np.sqrt(1-y[:,i]**2),y[:,i])
                if theta_y < 1 and theta_y > 0: 
                    F[I(i,j,M)] += 2 * bvp.mu / (h**2*(1+ theta_y)*theta_y) * bvp.g(x[j],np.sqrt(1-x[j]**2))
                    
    
    return F*h**2

def solve_bvp_circle(bvp, M, S):
    A = fdm_circle(bvp, M, S)
    F = rhs_circle(bvp,M, S)
    U = spsolve(A, F)
    return U

ex_1 = BVP(f2, v, u_ex2, 0, 1, 1, u_ex2)

# Number of subdivisions in each dimension
M = 40
S = shape_matrix(M)

#Define the grid using a sparse grid, and using the imaginary number 1j to include the endpoints
x,y = np.ogrid[0:1:(M+1)*1j, 0:1:(M+1)*1j]

# Evaluate u on the grid.
U_ext= ex_1.uexact(x, y)
for i in range(M+1):
    for j in range(M+1):
        if S[i,j] == 0: 
            U_ext[i,j] = 0
U_ext = U_ext.ravel()
U = solve_bvp_circle(ex_1, M, S)


print("Numerical sol:", U)
print("Exact sol:" ,U_ext)

error1 = U_ext-U
print(error1)
Eh = np.linalg.norm(error1,ord=np.inf)
print('The error is {:.2e}'.format(Eh))

plott(x,y,U.reshape((M+1,M+1)))
plott(x,y,U_ext.reshape((M+1,M+1)))
plott(x,y,error1.reshape((M+1,M+1)))

#error analysis
M_list=[10,20,40,80,160]
E=[]
h_list=[]
for M1 in M_list:
    x,y = np.ogrid[0:1:(M1+1)*1j, 0:1:(M1+1)*1j]
    S = shape_matrix(M1)
    U = solve_bvp_circle(ex_1, M1, S)
    U_ext= ex_1.uexact(x, y)
    for i in range(M1+1):
        for j in range(M1+1):
            if S[i,j] == 0: 
                U_ext[i,j] = 0
    U_ext = U_ext.ravel()
    error = U_ext - U
    E.append(np.linalg.norm(error,ord=np.inf))
    h_list.append(1/M1)

print(E)
print(h_list)
order = np.polyfit(np.log(h_list),np.log(E),1)[0]
print("order",order)

plt.figure()
plt.loglog(h_list,E,'o-')
plt.show()

