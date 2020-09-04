import pickle
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from gym.envs.mujoco.block2D import GOAL
from vices_garage.block2D import GOAL as GOAL_vices
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

plt.rcParams["figure.figsize"] = (6,2.5)
fig = plt.figure()
plt.axis('off')
# plt.rcParams['figure.constrained_layout.use'] = True

for i in range(len(task_list)):
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

    ax = fig.add_subplot(2, 2, 1)
    # ax.set_xlabel(r'Iteration')
    # ax.text(1, 1, r'Ours', fontweight="bold", rotation=90)
    plt.text(0.01, 0.8, '(a) Ours', fontweight='bold', fontsize=14, transform=plt.gcf().transFigure,rotation=90)
    plt.text(0.01, 0.45, '(b) VICES', fontweight='bold', fontsize=14, transform=plt.gcf().transFigure, rotation=90)
    ax.set_ylabel(r'Reward')
    ax.set_xlim(-50, 3050)
    ax.set_xticks([0,1000,2000,3000])
    ax.set_yticks([-100, -50, -0])
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    # ax.set_title(r'\textbf{(a)}', position=(-0.36, 0.85), fontsize=font_size_1)
    # ax.plot(rewards_undisc_mean, label='undisc. reward')
    inds = np.array(range(0,epoch_num)[1::10])*15
    heights = rewards_undisc_mean[1::10]
    yerr = rewards_undisc_std[1::10]
    yerr[0] = 0
    plt.errorbar(inds, heights, yerr=yerr, label='Task'+task_list[i],color=col[i])
    # ax.legend(frameon=False,prop={'size': font_size_2})
    ax.legend(loc='upper left', bbox_to_anchor=(-0.3, 1.35), frameon=False, ncol=3,prop={'size': font_size_2})

    ax = fig.add_subplot(2, 2, 2)
    ax.set_ylabel(r'Success \%')
    # ax.set_xlabel(r'Iteration')
    ax.set_xlim(-50, 3050)
    ax.set_xticks([0,1000,2000,3000])
    ax.set_yticks([0, 50,100])
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    # ax.set_title(r'\textbf{(b)}', position=(-0.3, 0.79), fontsize=font_size_1)
    width = 60
    # ax.plot(range(1,epoch_num+1)[::10], success_stat[::10])
    inds = np.array(range(1, epoch_num + 1)[i*3::10])*15
    success = success_stat[i*3::10]
    ax.bar(inds, success,width,color=col[i])
    # ax.legend(loc='upper left', bbox_to_anchor=(0., 1.25), frameon=False, ncol=3, prop={'size': font_size_2})
    plt.subplots_adjust(left=0.17, bottom=0.17, right=0.97, top=0.9, wspace=0.4, hspace=0.1)
# /home/shahbaz/Software/vices/vices_garage/data/local/exp/ppo_block_2d_task3_vices_cl0.1_ro30_cr10_itr100
base_filename = '/home/shahbaz/Software/vices/vices_garage/data/local/exp/'
file_list = ['ppo_block_2d_task1_vices_cl0.1_ro30_cr10_itr100',
             'ppo_block_2d_task2_vices_cl0.1_ro30_cr10_itr100',
             'ppo_block_2d_task3_vices_cl0.1_ro30_cr10_itr100']
task_list = ['1','2','3']
col = ['g','b','m']

for i in range(len(task_list)):
    filename = base_filename + file_list[i] + '/exp_log.pkl'
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
        # rewards_disc_rtn[ep] = np.mean([epoch[s]['returns'][0] for s in range(sample_num)])
        rewards_undisc_mean[ep] = np.mean([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
        rewards_undisc_std[ep] = np.std([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
        for s in range(sample_num):
            pos_norm = np.linalg.norm(epoch[s]['observations'][:, :2] - GOAL_vices, axis=1)
            success_mat[ep, s] = np.min(pos_norm)<SUCCESS_DIST

    success_stat = np.sum(success_mat, axis=1)*(100/sample_num)

    ax = fig.add_subplot(2, 2, 3)
    ax.set_xlabel(r'Rollouts')
    ax.set_ylabel(r'Reward')
    ax.set_xlim(-50, 3050)
    ax.set_xticks([0, 1000, 2000, 3000])
    ax.set_yticks([-100, -50, -0])
    # ax.set_title(r'\textbf{(b)}', position=(-0.36, 0.85), fontsize=font_size_1)
    # ax.xaxis.set_major_formatter(plt.NullFormatter())
    # ax.set_title(r'\textbf{(a)}', position=(-0.24, 0.79), fontsize=font_size_1)
    # ax.plot(rewards_undisc_mean, label='undisc. reward')
    # inds = range(1,epoch_num+1)[1::10]
    inds = np.array(range(1, epoch_num+1)[1::10]) * 30
    heights = rewards_undisc_mean[1::10]
    yerr = rewards_undisc_std[1::10]
    yerr[0] = 0
    plt.errorbar(inds, heights, yerr=yerr, label='Task'+task_list[i],color=col[i])
    # ax.legend(frameon=False,prop={'size': font_size_2})
    # ax.legend(loc='upper left', bbox_to_anchor=(-0.25, 1.4), frameon=False, ncol=3,prop={'size': font_size_2})
    ax = fig.add_subplot(2, 2, 4)
    ax.set_ylabel(r'Success \%')
    ax.set_xlabel(r'Rollouts')
    ax.set_xlim(-50, 3050)
    ax.set_xticks([0, 1000, 2000, 3000])
    ax.set_yticks([0, 50, 100])
    # ax.xaxis.set_major_formatter(plt.NullFormatter())
    # ax.set_title(r'\textbf{(b)}', position=(-0.3, 0.79), fontsize=font_size_1)
    width = 60
    # ax.plot(range(1,epoch_num+1)[::10], success_stat[::10])
    inds = np.array(range(0, epoch_num )[i*2::10]) * 30
    success = success_stat[(i+0)*2::10]
    ax.bar(inds, success,width,color=col[i])
    # ax.legend(loc='upper left', bbox_to_anchor=(0., 1.25), frameon=False, ncol=3, prop={'size': font_size_2})
    # plt.subplots_adjust(left=0.12, bottom=0.05, right=0.98, top=0.87, wspace=0.4, hspace=0.2)

# plt.show()
# fig.savefig("pos2_comparison.pdf", bbox_inches='tight',pad_inches=0.0)
fig.savefig("block_all_tasks_revise.pdf")
