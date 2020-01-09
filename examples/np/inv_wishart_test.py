import numpy as np
import matplotlib.pyplot as plt
from utils import plot_ellipse
from scipy.stats import invwishart

S_g_w = np.eye(2)

S_g = invwishart.rvs(4, S_g_w)

# S_g = np.array([
#     [1.1, .1],
#     [.7, 1.5]
# ])


def cost(S_g, samples):
    n = samples.shape[0]
    C = np.zeros((n))
    for i in range(n):
        s = samples[i] - S_g
        C[i] = np.square(np.linalg.norm(s))
    return C

# Sig = np.array([
#     [2, 0],
#     [0, 2]
# ])
# num_trials = 10
# dof = 1000
# for i in range(num_trials):
#     fig, ax = plt.subplots()
#     ax.set_xlim((-5, 5))
#     ax.set_ylim((-5, 5))
#     W = invwishart.rvs(dof, Sig)
#     plot_ellipse(ax, np.zeros(2), W*(dof-2-1))
#
# plt.show()


num_samples = 15
# num_samples = 50
num_itr = 30
percentage_best = 30
dof_init = 4
num_best = int(num_samples*percentage_best/100.0)

mean = np.array([0, 0])
# scale = .01
scale = 50
Sig_init_w = scale * np.eye(2)
"""
In KM book the scale matrix(Sig_init_w) is inverse and the mean is 
inv(Sig_init_w)/(dof-D-1)
The scipy implementation is different and hence the following
"""
Sig_init = invwishart.rvs(10, Sig_init_w)*(20-2-1)
Sig_curr = Sig_init
dof_curr = dof_init

plot_skip = 5
ax_lt = 20
# ax_lt = 1

for i in range(num_itr):
    samples = invwishart.rvs(dof_curr, Sig_curr, num_samples)*(dof_curr-2-1)
    Cs = cost(S_g, samples)
    best_inds = np.argsort(Cs)[:num_best]
    best_samples = samples[best_inds]
    avg_elite_cost = np.average(Cs[best_inds])
    Sig_curr = np.average(best_samples, axis=0)
    # dof_curr += int(np.absolute(num_best*avg_elite_cost))
    dof_curr += num_best
    print('Itr',i,':',avg_elite_cost, 'DOF:',dof_curr)
    if ((i+1)%plot_skip)==0:
        fig, ax = plt.subplots()
        ax.set_xlim((-ax_lt, ax_lt))
        ax.set_ylim((-ax_lt, ax_lt))
        plot_ellipse(ax, mean, Sig_init, color='b')
        plot_ellipse(ax, mean, S_g, color='r')
        plot_ellipse(ax, mean, Sig_curr, color='k')
        ax.set_title('Itr'+str(i))

plt.show()
