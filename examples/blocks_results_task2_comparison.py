import pickle
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from gym.envs.mujoco.block2D import GOAL
from utils import iMOGIC_energy_block_vec, iMOGIC_energy_blocks
from matplotlib import rc

plt.rcParams.update({'font.size': 12})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

############################################
'''
pos3 all failed.
show k16 1 for progress

pos2
k0 2, k1 3, k2 3, k4 4, k8 2, k16 4

'''
############################################

base_np_filename = '/home/shahbaz/Software/garage/examples/np/data/local'
prefix = 'blocks-initpos2-K'

K_list = ['0','1','2','4','8','16']
exp_name = ['2','3','3','4','2','4']
plt.rcParams["figure.figsize"] = (6,3)
fig = plt.figure()
plt.axis('off')
# plt.rcParams['figure.constrained_layout.use'] = True

for i in range(len(K_list)):
    filename = base_np_filename + '/' + prefix + K_list[i] + '/' + exp_name[i] + '/' + 'exp_log.pkl'
    infile = open(filename, 'rb')
    exp_log = pickle.load(infile)
    infile.close()

    filename = base_np_filename + '/' + prefix + K_list[i] + '/' + exp_name[i] + '/' + 'exp_param.pkl'
    infile = open(filename, 'rb')
    exp_param = pickle.load(infile)
    infile.close()

    epoch_num = len(exp_log)
    sample_num = len(exp_log[0])
    T = exp_log[0][0]['observations'].shape[0]
    tm = range(T)
    SUCCESS_DIST = 0.025

    rewards_disc_rtn = np.zeros(epoch_num)
    rewards_undisc_mean = np.zeros(epoch_num)
    rewards_undisc_std = np.zeros(epoch_num)
    success_mat = np.zeros((epoch_num, sample_num))
    for ep in range(epoch_num):
        epoch = exp_log[ep]
        rewards_disc_rtn[ep] = np.mean([epoch[s]['returns'][0] for s in range(sample_num)])
        rewards_undisc_mean[ep] = np.mean([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
        rewards_undisc_std[ep] = np.std([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
        for s in range(sample_num):
            pos_norm = np.linalg.norm(epoch[s]['observations'][:, :2] - GOAL, axis=1)
            success_mat[ep, s] = np.min(pos_norm)<SUCCESS_DIST

    success_stat = np.sum(success_mat, axis=1)*(100/sample_num)

    ax = fig.add_subplot(1, 2, 1)
    ax.set_xlabel(r'Iteration')
    ax.set_ylabel(r'Reward')
    # ax.plot(rewards_undisc_mean, label='undisc. reward')
    inds = range(1,epoch_num+1)[i::10]
    heights = rewards_undisc_mean[i::10]
    yerr = rewards_undisc_std[i::10]
    yerr[0] = 0
    plt.errorbar(inds, heights, yerr=yerr, label='K='+K_list[i])
    ax.legend()
    ax = fig.add_subplot(1, 2, 2)
    ax.set_ylabel(r'Succes rate')
    ax.set_xlabel(r'Iteration')
    width = 2
    # ax.plot(range(1,epoch_num+1)[::10], success_stat[::10])
    inds = range(1, epoch_num + 1)[i*2::15]
    success = success_stat[i*2::15]
    ax.bar(inds, success,width)
    plt.subplots_adjust(left=0.11, bottom=0.15, right=0.985, top=0.98, wspace=0.25, hspace=0.2)
plt.show()
# fig.savefig("pos2_comparison.pdf", bbox_inches='tight',pad_inches=0.0)
# fig.savefig("pos2_comparison.pdf")
