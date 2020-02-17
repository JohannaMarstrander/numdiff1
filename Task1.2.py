import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from Task1 import BVP,solve_bvp,plott


def u(x,y):
    return np.sin(np.pi*2*x) + np.cos(np.pi*2*y)
def f1(x,y):
    return (((2*np.pi)**2*np.sin(np.pi*2*x) + (2*np.pi)**2 * np.cos(np.pi*2*y))
           +v(x,y)[0]*(2*np.pi)*np.cos(np.pi*2*x) - v(x,y)[1]*(2*np.pi)*np.sin(np.pi*2*y))
def v(x,y):
    return np.array([x,y])


def u2(x,y):
    return np.exp(x) + 2*np.exp(y)
def f2(x,y):
    return -0.5*(np.exp(x)+2*np.exp(y)) + v2(x,y)[0]*np.exp(x) + 2*v2(x,y)[1]*np.exp(y) #K=1/& * (np.exp(1) + 1*np.exp(1)/my + 1*2*np.exp(1)/my
def v2(x,y):
    return np.array([2*x,y])

#error analysis
M_list=[10, 20, 39, 76, 150]
E=[]
h_list=[]
for M1 in M_list:
    x, y = np.ogrid[0:1:(M1 + 1) * 1j, 0:1:(M1 + 1) * 1j]
    ex = BVP(f2, v2, u2, 0, 1, 0.5, u2)
    U = solve_bvp(ex, M1)
    U_ext = ex.uexact(x, y).ravel()
    error = U_ext - U
    E.append(np.linalg.norm(error,ord=np.inf))
    h_list.append(1/M1)

print(E)
order = np.polyfit(np.log(h_list),np.log(E),1)[0]
print("order",order)

h_list=np.array(h_list)


plt.figure()
plt.loglog(h_list,E,'o-')
plt.loglog(h_list,h_list**2*1/12 * (np.exp(1) + 2*1*np.exp(1) + 1*2*np.exp(1)  ) )
plt.show()






