import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def I(i, j, n):
    return i + j * (n + 1)

#Tar bare inn indre gridpoints/altså de ukjente
def jacobi(U,M):
    h=1/(M+2)
    J = np.zeros(((M+1)*(M+1) ,(M+1)*(M+1)))

    for i in range(1,M):
        for j in range(1,M):
            J[I(i, j, M), I(i, j, M)] = -12 * U[I(i, j, M)] ** 2 +2*(U[I(i - 1, j, M)] + U[I(i, j - 1, M)] + U[I(i, j + 1, M)] + U[I(i + 1, j, M)])*U[I(i, j, M)]
            J[I(i, j, M), I(i + 1, j, M)] = U[I(i, j, M)] ** 2
            J[I(i, j, M), I(i -1 , j , M)] = U[I(i, j, M)] ** 2
            J[I(i, j, M), I(i, j - 1, M)] = U[I(i, j, M)] ** 2
            J[I(i, j, M), I(i, j + 1, M)] = U[I(i, j, M)] ** 2

    #Langs i=0 og I=M, ikke hjørner
    for j in range(1, M):
        J[I(0, j, M), I(0, j, M)] = -12*U[I(0,j,M)] ** 2 +2*(1 + U[I(0, j - 1, M)] + U[I(0, j + 1, M)] + U[I(0, j, M)])*U[I(0, j, M)]
        J[I(0, j, M), I(0 + 1, j, M)] = U[I(0,j,M)] ** 2
        J[I(0, j, M), I(0, j - 1, M)] = U[I(0,j,M)] ** 2
        J[I(0, j, M), I(0, j + 1, M)] = U[I(0,j,M)] ** 2

        J[I(M, j, M), I(M, j, M)] = -12*U[I(M,j,M)] ** 2 +2*( U[I(M, j - 1, M)] + U[I(M, j + 1, M)] + U[
            I(M-11, j, M)] + 1) * U[I(M, j, M)]
        J[I(M, j, M), I(M - 1, j, M)] = U[I(M,j,M)] ** 2
        J[I(M, j, M), I(M, j - 1, M)] = U[I(M,j,M)] ** 2
        J[I(M, j, M), I(M, j + 1, M)] = U[I(M,j,M)] ** 2

    #Langs j=0 og j=M
    for i in range(1, M):
        J[I(i, 0, M), I(i, 0, M)] = -12 *  U[I(i, 0, M)] ** 2 +2*( U[I(i - 1, 0, M)]  + U[I(i,  1, M)] + U[
            I(i + 1, 0, M)]+1) * U[I(i, 0, M)]
        J[I(i, 0, M), I(i + 1, 0, M)] =  U[I(i, 0, M)] ** 2
        J[I(i, 0, M), I(i-1, 0, M)] =  U[I(i, 0, M)] ** 2
        J[I(i, 0, M), I(i, 0 + 1, M)] = U[I(i, 0, M)] ** 2

        J[I(i, M, M), I(i, M, M)] = -12 * U[I(i, M, M)] ** 2 + 2*(U[I(i - 1, M, M)]  + U[I(i, M - 1, M)] + U[
            I(i + 1, M, M)]+1) * U[I(i, M, M)]
        J[I(i, M, M), I(i + 1, M, M)] =  U[I(i, M, M)] ** 2
        J[I(i, M, M), I(i-1, M, M)] =  U[I(i, M, M)] ** 2
        J[I(i, M, M), I(i, M-1, M)] = U[I(i, M, M)] ** 2

    #hjørner
    J[I(0, 0, M), I(0, 0, M)] = -12 *  U[I(0, 0, M)] ** 2 + 2*(U[I(1,0,M)]+ U[I(0,1,M)] + 2)*U[I(0,0,M)]
    J[I(0, 0, M), I(1, 0, M)] = U[I(0, 0, M)] ** 2
    J[I(0, 0, M), I(0, 1, M)] = U[I(0, 0, M)] ** 2

    J[I(0, M, M), I(0, M, M)] = -12 *  U[I(0, M, M)] ** 2 + 2*(+ U[I(1, M, M)] + U[I(0, M-1, M)] + 2)*U[I(0,M,M)]
    J[I(0, M, M), I(0, M-1, M)] = U[I(0, M, M)] ** 2
    J[I(0, M, M), I(1, M, M)] = U[I(0, M, M)] ** 2

    J[I(M, 0, M), I(M, 0, M)] = -12 *  U[I(M, 0, M)] ** 2 + 2*(U[I(M-1,0,M)] + U[I(M,1,M)] + 2)*U[I(M,0,M)]
    J[I(M, 0, M), I(M, 1, M)] =  U[I(M, 0, M)] ** 2
    J[I(M, 0, M), I(M-1, 0, M)] =   U[I(M, 0, M)] ** 2

    J[I(M, M, M), I(M, M, M)] = -12 *  U[I(M, M, M)] ** 2 + 2*(U[I(M-1, M, M)] + U[I(M, M-1, M)] + 2)*U[I(M,M,M)]
    J[I(M, M, M), I(M-1, M, M)] = U[I(M, M, M)] ** 2
    J[I(M, M, M), I(M , M-1, M)] = U[I(M, M, M)] ** 2

    return J/(h*h)

def G1(U,M):
    G = np.zeros((M+1)*(M+1))
    lamb = 1.5
    h = 1/M
    hh= h*h
    #hjørner
    #print(U[I(0,1,M)])
    G[I(0,0,M)] = (-4*U[I(0,0,M)] + U[I(1,0,M)]+ U[I(0,1,M)] + 2)*(U[I(0,0,M)]**2)*1/hh -lamb #hardkodet boundary
    G[I(M,0,M)] = (-4*U[I(M,0,M)] + U[I(M-1,0,M)] + U[I(M,1,M)] + 2)*(U[I(M,0,M)]**2)*1/hh -lamb
    G[I(0, M, M)] = (-4 * U[I(0, M, M)]  + U[I(1, M, M)] + U[I(0, M-1, M)] + 2)*(U[I(0,M,M)]**2)*1/hh -lamb
    G[I(M,M,M)] = (-4 * U[I(M, M, M)]  + U[I(M-1, M, M)] + U[I(M, M-1, M)] + 2)*(U[I(M,M,M)]**2)*1/hh -lamb


    #kanter
    for i in range(1,M):
        G[I(i, 0, M)] = (-4 * U[I(i, 0, M)] + U[I(i - 1, 0, M)]  + U[I(i,  1, M)] + U[
            I(i + 1, 0, M)]+1 ) * U[I(i, 0, M)] ** 2 * 1 / hh - lamb
        G[I(i, M, M)] = (-4 * U[I(i, M, M)] + U[I(i - 1, M, M)]  + U[I(i, M - 1, M)] + U[
            I(i + 1, M, M)]+1 ) * (U[I(i, M, M)] ** 2) * 1 / hh - lamb

    for j in range(1,M):
        G[I(0, j, M)] = (-4 * U[I(0, j, M)] + U[I(0, j - 1, M)] + U[I(0, j + 1, M)] + U[
            I(1, j, M)]+1) * (U[I(0, j, M)] ** 2 )* 1 / hh - lamb
        G[I(M, j, M)] = (-4 * U[I(M, j, M)] + U[I(M, j - 1, M)] + U[I(M, j + 1, M)] + U[
            I(M-1, j, M)] + 1) *( U[I(M, j, M)] ** 2 )* 1 / hh - lamb


    for i in range(1,M):
        for j in range(1,M):
            G[I(i, j, M)] = (-4 * U[I(i, j, M)] + U[I(i - 1, j, M)] + U[I(i, j - 1, M)] + U[I(i, j + 1, M)] + U[I(i + 1, j, M)])*(U[I(i, j, M)]**2)*1/hh - lamb
    #print(G)
    return G

M=60
err=10
tol=0.0000001
it=0
iter_max=100
u_k1=np.linspace(0.9,0.95,(M+1)*(M+1))
F=G1(u_k1,M)

print("Newton")
while (err>tol and it<iter_max):
    J=jacobi(u_k1,M)
    a = np.linalg.solve(J,F)
    u_k1 -= a
    F = G1(u_k1, M)
    err = la.norm(F)
    it+=1

def plott(x,y,Z):

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
    #ax.view_init(30,225)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()




print(tol,iter_max)
print("err",err,"it",it)



x, y = np.ogrid[0:1:(M + 1) * 1j, 0:1:(M + 1) * 1j]

plott(x,y,u_k1.reshape((M+1),(M+1)))


x, y =np.ogrid[0:1:(M + 3) * 1j, 0:1:(M + 3) * 1j]


I2=np.ones(M+1)
A=np.vstack((I2,u_k1.reshape((M+1),(M+1)),I2))
I=np.ones(M+3)
I=I[:, np.newaxis]
U=np.hstack((I,A,I))
plott(x,y,U)



#U=[1,2,3,4,5,6,7,8,9]
#U2=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
#U2=np.ones(16)
#print(len(U))

#print(G(U,2))
#
#print(G1(U2,3))

