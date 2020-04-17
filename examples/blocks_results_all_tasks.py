import pickle
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from gym.envs.mujoco.block2D import GOAL
from utils import iMOGIC_energy_block_vec, iMOGIC_energy_blocks
from matplotlib import rc

font_size_1 = 12
font_size_2 = 10
plt.rcParams.update({'font.size': font_size_1})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

############################################
'''
task3 all failed.
k16-1

task2
k8-2

task1
k1-4

'''
############################################

base_np_filename = '/home/shahbaz/Software/garage/examples/np/data/local/blocks-initpos'
prefix = 'blocks-initpos2-K'
task_list = ['1','2','3']
# task_list = ['3','2','1']
K_list = ['1','8','16']
exp_name = ['4','2','1']
col = ['g','b','m']

plt.rcParams["figure.figsize"] = (6,2)
fig = plt.figure()
plt.axis('off')
# plt.rcParams['figure.constrained_layout.use'] = True

for i in range(len(K_list)):
    filename = base_np_filename + task_list[i] + '-K' + K_list[i] + '/' + exp_name[i] + '/' + 'exp_log.pkl'
    infile = open(filename, 'rb')
    exp_log = pickle.load(infile)
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
    # ax.set_title(r'\textbf{(a)}', position=(-0.24, 0.79), fontsize=font_size_1)
    # ax.plot(rewards_undisc_mean, label='undisc. reward')
    inds = range(1,epoch_num+1)[1::10]
    heights = rewards_undisc_mean[1::10]
    yerr = rewards_undisc_std[1::10]
    yerr[0] = 0
    plt.errorbar(inds, heights, yerr=yerr, label='Task'+task_list[i],color=col[i])
    ax.legend(frameon=False,prop={'size': font_size_2})
    ax = fig.add_subplot(1, 2, 2)
    ax.set_ylabel(r'Success rate')
    ax.set_xlabel(r'Iteration')
    # ax.set_title(r'\textbf{(b)}', position=(-0.3, 0.79), fontsize=font_size_1)
    width = 2
    # ax.plot(range(1,epoch_num+1)[::10], success_stat[::10])
    inds = range(1, epoch_num + 1)[i*2::10]
    success = success_stat[i*2::10]
    ax.bar(inds, success,width,color=col[i])
    plt.subplots_adjust(left=0.12, bottom=0.22, right=0.98, top=0.98, wspace=0.4, hspace=0.2)
# plt.show()
# fig.savefig("pos2_comparison.pdf", bbox_inches='tight',pad_inches=0.0)
fig.savefig("block_all_tasks.pdf")
