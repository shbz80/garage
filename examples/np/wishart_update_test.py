import numpy as np
import matplotlib.pyplot as plt
# from utils import plot_ellipse
from scipy.stats import wishart
import random

num_trials = 10
random.seed(7)
plt.rcParams.update({'font.size': 18})

for i in range(num_trials):
    # D = 7
    D = np.random.randint(2, 10)
    dofs = np.logspace(D+2, 16., num=10, base=np.exp(1), dtype=np.uint64)

    S_g_w = 10*np.eye(D)
    # S_g = S_g_w
    S_g = wishart.rvs(D+2, S_g_w)
    # S_g = np.array([
    #     [1.1, .1],
    #     [.7, 1.5]
    # ])
    # S_g = np.array([
    #         [1.5]])

    entropy = np.zeros(len(dofs))
    std = np.zeros(len(dofs))
    for i in range(len(dofs)):
        S = S_g/dofs[i]
        # S = S_g
        v = dofs[i]
        entropy[i] = wishart.entropy(v, S)
        # S_sq = np.square(S)
        # sig_ii = np.tile(np.diag(S).reshape(-1,1), (1,D))
        # sig_jj = np.tile(np.diag(S), (D,1))
        # var_mat = v*(S_sq + np.multiply(sig_ii, sig_jj))
        # # var_mat = (np.square(S) + np.multiply(np.tile(np.diag(S), (D,1)),np.tile(np.diag(S).reshape(-1,1), (1,D))))*v
        # std[i] = np.sum(var_mat.flatten())

    num_trials = 10
    # random.seed(4)

    for i in range(num_trials):
        D = 7
        # D = np.random.randint(2, 10)
        dofs = np.logspace(D + 2, 16., num=10, base=np.exp(1), dtype=np.uint64)

        S_g_w = 10 * np.eye(D)
        # S_g = S_g_w
        S_g = wishart.rvs(D + 2, S_g_w)
        # S_g = np.array([
        #     [1.1, .1],
        #     [.7, 1.5]
        # ])
        # S_g = np.array([
        #         [1.5]])

        entropy_1 = np.zeros(len(dofs))
        std = np.zeros(len(dofs))
        for i in range(len(dofs)):
            S = S_g / dofs[i]
            # S = S_g
            v = dofs[i]
            entropy_1[i] = wishart.entropy(v, S)
            # S_sq = np.square(S)
            # sig_ii = np.tile(np.diag(S).reshape(-1,1), (1,D))
            # sig_jj = np.tile(np.diag(S), (D,1))
            # var_mat = v*(S_sq + np.multiply(sig_ii, sig_jj))
            # # var_mat = (np.square(S) + np.multiply(np.tile(np.diag(S), (D,1)),np.tile(np.diag(S).reshape(-1,1), (1,D))))*v
            # std[i] = np.sum(var_mat.flatten())

    # plt.figure()

    ax = plt.subplot(1,2,1)
    ax.plot(entropy, np.log(dofs))
    ax.set_xlabel(r'$H$')
    ax.set_ylabel(r'$ln\nu$')

    ax = plt.subplot(1, 2, 2)
    ax.plot(entropy_1, np.log(dofs))
    ax.set_xlabel(r'$H$')
    ax.set_ylabel(r'$ln\nu$')

    # ax = plt.subplot(2, 3, 1)
    # ax.plot(entropy, dofs)
    # ax.set_xlabel('H')
    # ax.set_ylabel('v')

    # ax = plt.subplot(2,3,2)
    # ax.plot(entropy, np.log(std))
    # ax.set_xlabel('H')
    # ax.set_ylabel('log std')

    # ax = plt.subplot(2, 3, 2)
    # ax.plot(entropy, std)
    # ax.set_xlabel('H')
    # ax.set_ylabel('std')

    # ax = plt.subplot(2,3,3)
    # ax.plot(np.log(std), np.log(dofs))
    # ax.set_xlabel('log std')
    # ax.set_ylabel('log v')
    #
    # ax = plt.subplot(2, 3, 4)
    # ax.plot(var_mat.flatten())
    # ax.set_title('var_mat')
    #
    # ax = plt.subplot(2, 3, 5)
    # ax.plot(np.sqrt(var_mat.flatten()))
    # ax.set_title('std_mat')

plt.show()
