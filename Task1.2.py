import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from Task1 import BVP,solve_bvp,plott


def u(x,y):
    return np.sin(np.pi*2*x) + np.cos(np.pi*2*y)
def f1(x,y):
    return (0.01*((2*np.pi)**2*np.sin(np.pi*2*x) + (2*np.pi)**2 * np.cos(np.pi*2*y))
           +v(x,y)[0]*(2*np.pi)*np.cos(np.pi*2*x) - v(x,y)[1]*(2*np.pi)*np.sin(np.pi*2*y))
def v(x,y):
    return np.array([x,-2*y])


#error analysis
M_list=[5,10,20,40]
E=[]
h_list=[]
for M1 in M_list:
    x, y = np.ogrid[0:1:(M1 + 1) * 1j, 0:1:(M1 + 1) * 1j]
    ex = BVP(f1, v, u, 0, 1, 0.01, u)
    U = solve_bvp(ex, M1)
    U_ext = ex.uexact(x, y).ravel()
    error = U_ext - U
    E.append(np.linalg.norm(error,ord=np.inf))
    h_list.append(1/M1)

print(E)
order = np.polyfit(np.log(h_list),np.log(E),1)[0]
print("order",order)

plt.loglog(h_list,E,'o-')
plt.show()

plott(x,y,u(x,y))

U_ny=U.reshape((M1+1,M1+1))
plott(x,y,U_ny)

