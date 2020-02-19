import numpy as np
import matplotlib.pyplot as plt
from Task1_a import BVP, solve_bvp, plott


def u(x, y):
    return np.sin(np.pi * 1 / 2 * x) + np.cos(np.pi * y)


def f1(x, y):
    return (((1 / 4 * np.pi ** 2) * np.sin(np.pi * 1 / 2 * x) + (np.pi) ** 2 * np.cos(np.pi * y))
            + v(x, y)[0] * (1 / 2 * np.pi) * np.cos(np.pi * 1 / 2 * x) - v(x, y)[1] * (np.pi) * np.sin(np.pi * y))


def v(x, y):
    return np.array([x, y])


# error analysis
M_list = [10, 20, 39, 76]  # 150
E = []
h_list = []

for M1 in M_list:
    x, y = np.ogrid[0:1:(M1 + 1) * 1j, 0:1:(M1 + 1) * 1j]
    ex = BVP(f1, v, u, 0, 1, 1, u)
    U = solve_bvp(ex, M1)
    U_ext = ex.uexact(x, y)
    if M1 == 150:
        plott(x, y, U.reshape((M1 + 1, M1 + 1)))
        plott(x, y, U_ext)
    U_ext = U_ext.ravel()
    error = U_ext - U
    E.append(np.linalg.norm(error, ord=np.inf))
    h_list.append(1 / M1)

print(E)
order = np.polyfit(np.log(h_list), np.log(E), 1)[0]
print("order", order)
h_list = np.array(h_list)

plt.figure()
plt.loglog(h_list, E, 'o-')
print((1 / 12) * ((1) * np.pi ** 4 + (1 + 1 / 8) * np.pi ** 3))
plt.loglog(h_list, h_list ** 2 * 1 / 12 * ((1 + 1 / 16) * np.pi ** 4 + 18 * np.pi ** 3))
plt.loglog(h_list, h_list ** 2 * 1 / 12 * (np.pi ** 4 + (1 + 1 / 8) * np.pi ** 3), color='r')
plt.show()
