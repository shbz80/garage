from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

s1_l = 50
s2_l = 50

s1 = np.linspace(-10, 20, s1_l)
s2 = np.linspace(-20, 10, s2_l)

l = 1*0.5
l = l*.001
S1, S2 = np.meshgrid(s1, s2, indexing='ij')

def meshplot(S_k, mu_k):
    A = np.zeros((s1_l, s2_l))
    B = np.zeros((s1_l, s2_l))
    AB = np.zeros((s1_l, s2_l))
    V = np.zeros((s1_l, s2_l))
    F_AB = np.zeros((s1_l, s2_l, 2))
    F = np.zeros((s1_l, s2_l, 2))
    for i in range(s1_l):
        for j in range(s2_l):
            s = np.array([S1[i,j], S2[i,j]])
            Sk_dot_mu = S_k.dot(s-2.*mu_k)
            v = s.dot(Sk_dot_mu)
            if v>=0:
                A[i,j] = s.dot(Sk_dot_mu)
                e = -l*0.25*A[i,j]**2
                B[i,j] = np.exp(e)
                AB[i,j] = A[i,j]*B[i,j]
                V[i,j] = 1/l * (1. - B[i,j])
                F_AB[i,j] = -AB[i,j]*S_k.dot(s-mu_k)
            else:
                A[i, j] = 0.
                B[i, j] = 1.
                AB[i, j] =0.
                V[i, j] = 0.
                F_AB[i, j] = np.zeros(2)
            F[i, j] = -S_k.dot(s - mu_k)
    return A, B, AB, V, F_AB, F

def _meshplot(S_k, mu_k):
    A = np.zeros((s1_l, s2_l))
    B = np.zeros((s1_l, s2_l))
    AB = np.zeros((s1_l, s2_l))
    V = np.zeros((s1_l, s2_l))
    F_AB = np.zeros((s1_l, s2_l, 2))
    F = np.zeros((s1_l, s2_l, 2))
    for i in range(s1_l):
        for j in range(s2_l):
            s = np.array([S1[i,j], S2[i,j]])
            Sk_dot_mu = S_k.dot(s-2.*mu_k)
            A[i,j] = (s-2.*mu_k).dot(Sk_dot_mu)
            e = -l * 0.25 * A[i, j]**2
            B[i, j] = np.exp(e)
            AB[i, j] = A[i, j] * B[i, j]
            V[i, j] = 1 / l * (1. - B[i, j])
            F_AB[i, j] = -AB[i, j] * S_k.dot(s - mu_k)
            F[i, j] = -S_k.dot(s - mu_k)
    return A, B, AB, V, F_AB, F

S_k_1 = np.array([
    [1.1, .1],
    [.7, 1.5]
])
mu_k_1 = np.array([4, -3])
# mu_k_1 = np.array([10, -5])
print(np.linalg.eigvals(S_k_1))
assert(np.all(np.linalg.eigvals(S_k_1) > 0))

S_k_0 = np.array([
    [2., 0],
    [0, 1.]
])
mu_k_0 = np.array([0, 0])
print(np.linalg.eigvals(S_k_0))
assert(np.all(np.linalg.eigvals(S_k_0) > 0))



A1, B1, AB1, V1, FAB1, F1 = meshplot(S_k_1, mu_k_1)
F1_1 = F1[:, :, 0]
F1_2= F1[:, :, 1]
FAB1_1 = FAB1[:, :, 0]
FAB1_2= FAB1[:, :, 1]

A0, B0, AB0, V0, FAB0, F0 = meshplot(S_k_0, mu_k_0)
F0_1 = F0[:, :, 0]
F0_2 = F0[:, :, 1]
FAB0_1 = FAB0[:, :, 0]
FAB0_2 = FAB0[:, :, 1]

fig = plt.figure()
ax = fig.add_subplot(2,4,1,projection="3d")
ax.plot_surface(S1, S2, A1, cmap='viridis', edgecolor='none')
# ax.set_title('Alpha-original')
ax.set_xlabel('s1')
ax.set_ylabel('s2')
ax.set_zlabel(r'$\alpha_1$')
ax = fig.add_subplot(2,4,2,projection="3d")
ax.plot_surface(S1, S2, B1, cmap='viridis', edgecolor='none')
# ax.set_title('Beta-original')
ax.set_xlabel('s1')
ax.set_ylabel('s2')
ax.set_zlabel(r'$\beta_1$')
ax = fig.add_subplot(2,4,3,projection="3d")
ax.plot_surface(S1, S2, AB1, cmap='viridis', edgecolor='none')
# ax.set_title('Alpha*Beta-original')
ax.set_xlabel('s1')
ax.set_ylabel('s2')
ax.set_zlabel(r'$w_1 = \alpha_1.\beta_1$')
ax = fig.add_subplot(2,4,4,projection="3d")
ax.plot_surface(S1, S2, V1, cmap='viridis', edgecolor='none')
# ax.set_title('Potential-original')
ax.set_xlabel('s1')
ax.set_ylabel('s2')
ax.set_zlabel(r'$\phi_1$')

ax = fig.add_subplot(2,4,5,projection="3d")
ax.plot_surface(S1, S2, A0, cmap='viridis', edgecolor='none')
# ax.set_title('Alpha-test')
ax.set_xlabel('s1')
ax.set_ylabel('s2')
ax.set_zlabel(r'$\alpha_0$')
ax = fig.add_subplot(2,4,6,projection="3d")
ax.plot_surface(S1, S2, B0, cmap='viridis', edgecolor='none')
# ax.set_title('Beta-test')
ax.set_xlabel('s1')
ax.set_ylabel('s2')
ax.set_zlabel(r'$\beta_0$')
ax = fig.add_subplot(2,4,7,projection="3d")
ax.plot_surface(S1, S2, AB0, cmap='viridis', edgecolor='none')
# ax.set_title('Alpha*Beta-test')
ax.set_xlabel('s1')
ax.set_ylabel('s2')
ax.set_zlabel(r'$w_0 = \alpha_0.\beta_0$')
ax = fig.add_subplot(2,4,8,projection="3d")
ax.plot_surface(S1, S2, V0, cmap='viridis', edgecolor='none')
# ax.set_title('Potential-test')
ax.set_xlabel('s1')
ax.set_ylabel('s2')
ax.set_zlabel(r'$\phi_0$')

fig = plt.figure()
ax = fig.add_subplot(121, projection="3d")
ax.plot_surface(S1, S2, V0+V1, cmap='viridis', edgecolor='none')
# ax.set_title('Potential-test')
ax.set_xlabel('s1')
ax.set_ylabel('s2')
ax.set_zlabel(r'$\Phi=\phi_0+\phi_1$')
ax = fig.add_subplot(122, projection="3d")
ax.plot_surface(S1, S2, V0+V1+0.5*A0, cmap='viridis', edgecolor='none')
# ax.set_title('Potential-test')
ax.set_xlabel('s1')
ax.set_ylabel('s2')
ax.set_zlabel(r'$\Phi=\phi_*+\phi_0+\phi_1$')

fig = plt.figure()
ax = fig.add_subplot(2,3,1)
ax.quiver(S1, S2, F1_1, F1_2, color='b')
ax.set_title(r'$f_1^{stiff}$')
ax.set_xlabel('s1')
ax.set_ylabel('s2')
ax = fig.add_subplot(2,3,4)
ax.quiver(S1, S2, FAB1_1, FAB1_2, color='b')
ax.set_title(r'$w_1.f_1^{stiff}$')
ax.set_xlabel('s1')
ax.set_ylabel('s2')
ax = fig.add_subplot(2,3,2)
ax.quiver(S1, S2, F0_1, F0_2, color='b')
ax.set_title(r'$f_0^{stiff}$')
ax.set_xlabel('s1')
ax.set_ylabel('s2')
ax = fig.add_subplot(2,3,5)
ax.quiver(S1, S2, FAB0_1, FAB0_2, color='b')
ax.set_title(r'$w_0.f_0^{stiff}$')
ax.set_xlabel('s1')
ax.set_ylabel('s2')
ax = fig.add_subplot(2,3,6)
ax.quiver(S1, S2, FAB0_1+FAB1_1, FAB0_2+FAB1_2, color='b')
ax.set_title(r'$w_0.f_0^{stiff}+w_1.f_1^{stiff}$')
ax.set_xlabel('s1')
ax.set_ylabel('s2')
ax = fig.add_subplot(2,3,3)
ax.quiver(S1, S2, FAB0_1+FAB1_1+F0_1, FAB0_2+FAB1_2+F0_2, color='b')
# ax.quiver(S1, S2, F0_1+FAB0_1*0.1+FAB1_1*0.1, F0_2+FAB0_2*0.1+FAB1_2*0.1)
ax.set_title(r'$f_*^{stiff}+w_0.f_0^{stiff}+w_1.f_1^{stiff}$')
ax.set_xlabel('s1')
ax.set_ylabel('s2')

plt.show()
