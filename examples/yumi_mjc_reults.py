import pickle
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from gym.envs.mujoco.yumipeg import GOAL
from YumiKinematics import YumiKinematics
from utils import iMOGIC_energy_yumi_vec, iMOGIC_energy_yumi
from matplotlib import rc

font_size_1 = 12
font_size_2 = 14
font_size_3 = 10
plt.rcParams.update({'font.size': font_size_1})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


data_file_peg_woinit = '/home/shahbaz/Software/garage/examples/np/data/local/peg-woinit-full-imped/1/exp_log.pkl'
data_file_peg_winit = '/home/shahbaz/Software/garage/examples/np/data/local/peg-winit-full-imped/2/exp_log.pkl'
data_file_peg_rnd_50 = '/home/shahbaz/Software/garage/examples/np/data/local/peg-random-init-pos-itr50/3/exp_log.pkl'
data_file_peg_rnd_0 = '/home/shahbaz/Software/garage/examples/np/data/local/peg-random-init-pos-itr0/1/exp_log.pkl'
param_file_peg_woinit = '/home/shahbaz/Software/garage/examples/np/data/local/peg-woinit-full-imped/1/exp_param.pkl'
param_file_peg_winit = '/home/shahbaz/Software/garage/examples/np/data/local/peg-winit-full-imped/2/exp_param.pkl'

infile = open(data_file_peg_winit, 'rb')
peg_winit_data = pickle.load(infile)
infile.close()

infile = open(data_file_peg_rnd_50, 'rb')
peg_rnd_50_data = pickle.load(infile)
infile.close()

infile = open(data_file_peg_rnd_0, 'rb')
peg_rnd_0_data = pickle.load(infile)
infile.close()

infile = open(data_file_peg_woinit, 'rb')
peg_woinit_data = pickle.load(infile)
infile.close()

infile = open(param_file_peg_winit, 'rb')
peg_winit_param = pickle.load(infile)
infile.close()

infile = open(param_file_peg_woinit, 'rb')
peg_woinit_param = pickle.load(infile)
infile.close()

yumikinparams = {}
yumikinparams['urdf'] = '/home/shahbaz/Software/yumi_kinematics/yumikin/models/yumi_ABB_left.urdf'
yumikinparams['base_link'] = 'world'
# yumikinparams['end_link'] = 'gripper_l_base'
# yumikinparams['end_link'] = 'left_tool0'
yumikinparams['end_link'] = 'left_contact_point'
yumikinparams['euler_string'] = 'sxyz'
yumikinparams['goal'] = GOAL
yumiKin = YumiKinematics(yumikinparams)
GOAL = yumiKin.goal_cart

epoch_num = len(peg_winit_data)
sample_num = len(peg_winit_data[0])
T = peg_winit_data[0][0]['observations'].shape[0]
tm = range(T)
SUCCESS_DIST = .01
K=2
dS = 6

rewards_undisc_mean = np.zeros(epoch_num)
rewards_undisc_std = np.zeros(epoch_num)
success_mat = np.zeros((epoch_num, sample_num))
Vw = np.zeros(epoch_num)
Vwo = np.zeros(epoch_num)
# S_array_w = np.zeros((sample_num,dS**2))
# S_array_wo = np.zeros((sample_num,dS**2))
# S_array_w = np.zeros((sample_num,dS*K))
# S_array_wo = np.zeros((sample_num,dS*K))
mu_array = np.zeros((sample_num,dS*K))
l_array = np.zeros((sample_num,K))
mu_std = np.zeros(epoch_num)
l_std = np.zeros(epoch_num)

for ep in range(epoch_num):
    epoch = peg_winit_data[ep]
    rewards_undisc_mean[ep] = np.mean([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
    rewards_undisc_std[ep] = np.std([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
    for s in range(sample_num):
        sample = np.min(np.absolute(epoch[s]['observations'][:,:3]), axis=0)
        # sample = epoch[s]['observations'][:, :6]
        success_mat[ep, s] = np.linalg.norm(sample) < SUCCESS_DIST
    ep_param_w = peg_winit_param[ep]
    Vw[ep] = ep_param_w['epoc_stat']['sd_dof']['base'][0]['S']
    ep_param_wo = peg_woinit_param[ep]
    Vwo[ep] = ep_param_wo['epoc_stat']['sd_dof']['base'][0]['S']
    for i in range(sample_num):
        mu_array[i] = ep_param_w['epoc_params'][i][0]
        l_array[i] = ep_param_w['epoc_params'][i][2]
    mu_std[ep] = np.mean(np.std(mu_array, axis=0))
    l_std[ep] = np.mean(np.std(l_array, axis=0))

success_stat = np.sum(success_mat, axis=1)*(100/sample_num)

plt.rcParams["figure.figsize"] = (6,2)

interval = 5
idx = 0
fig1 = plt.figure()
plt.axis('off')
ax1 = fig1.add_subplot(1, 2, 1)
ax1.set_title(r'\textbf{(a)}',position=(-0.3,0.97), fontsize=font_size_2)
ax1.set_xlabel(r'Iteration')
ax1.set_ylabel(r'Reward')
ax1.set_xticks(range(0,100,20))
inds = range(1,epoch_num+1)[idx::interval]
heights = rewards_undisc_mean[idx::interval]
yerr = rewards_undisc_std[idx::interval]
yerr[0] = 0
ax1.errorbar(inds, heights, yerr=yerr, label=r'with init', color='b')
ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,00))
ax1.legend(prop={'size': font_size_3},frameon=False)
ax2 = fig1.add_subplot(1, 2, 2)
ax2.set_ylabel(r'Success rate')
ax2.set_xlabel('Iteration')
ax2.set_title(r'\textbf{(b)}',position=(-0.32,0.97), fontsize=font_size_2)
ax2.set_xticks(range(0,100,20))
width = 2
inds = range(1, epoch_num + 1)[idx::interval]
success = success_stat[idx::interval]
ax2.bar(inds, success,width, color='b')
ax2.legend(prop={'size': font_size_3},frameon=False)
epoch_num = len(peg_woinit_data)
sample_num = len(peg_woinit_data[0])
T = peg_woinit_data[0][0]['observations'].shape[0]
tm = range(T)

rewards_undisc_mean = np.zeros(epoch_num)
rewards_undisc_std = np.zeros(epoch_num)
success_mat = np.zeros((epoch_num, sample_num))
for ep in range(epoch_num):
    epoch = peg_woinit_data[ep]
    rewards_undisc_mean[ep] = np.mean([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
    rewards_undisc_std[ep] = np.std([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
    for s in range(sample_num):
        # sample = np.min(np.absolute(epoch[s]['observations'][:,:3]), axis=0)
        sample = np.min(np.linalg.norm(epoch[s]['observations'][:, :3],axis=1), axis=0)
        # sample = epoch[s]['observations'][:, :6]
        # success_mat[ep, s] = np.linalg.norm(sample) < SUCCESS_DIST
        success_mat[ep, s] = sample < SUCCESS_DIST
success_stat = np.sum(success_mat, axis=1)*(100/sample_num)

interval = 5
idx = 1
# ax1 = fig.add_subplot(2, 2, 1)
inds = range(1,epoch_num+1)[:100][idx::interval]
heights = rewards_undisc_mean[:100][idx::interval]
yerr = rewards_undisc_std[:100][idx::interval]
yerr[0] = 0
ax1.errorbar(inds, heights, yerr=yerr, label=r'w/o init', color='g')
ax1.legend(prop={'size': font_size_3},frameon=False)
# ax2 = fig.add_subplot(2, 2, 2)
width = 2
inds = range(1, epoch_num + 1)[:100][idx*2::interval]
success = success_stat[:100][idx*2::interval]
ax2.bar(inds, success,width, color='g')
plt.subplots_adjust(left=0.13, bottom=0.22, right=.99, top=0.87, wspace=0.5, hspace=0.7)
# plt.show(block=False)
fig1.savefig("yumi_mjc_progress.pdf")

plt.rcParams["figure.figsize"] = (6,1.35)
fig2 = plt.figure()
# plt.axis('off')
ax3 = fig2.add_subplot(1, 2, 1)
ax3.set_ylabel(r'$ln\ \nu$')
ax3.set_xlabel('Iteration')
ax3.set_title(r'\textbf{(c)}',position=(-0.3,0.94), fontsize=font_size_2)
ax3.set_xticks(range(0,100,20))
# ax3.plot(np.log(Vw), label=r'with init', color='b')
ax3.plot(np.log(Vw), color='b')
# ax3.plot(np.log(Vwo), label=r'w/o init', color='g')

# ax3.set_ylim(-5)
ax3.legend(loc='upper left', bbox_to_anchor=(0.5, 0.6), frameon=False, ncol=1, prop={'size': font_size_3})


ax4 = fig2.add_subplot(1, 2, 2)
ax4.set_ylabel(r'S. D.')
ax4.set_xlabel('Iteration')
ax4.set_title(r'\textbf{(d)}',position=(-0.25,0.94), fontsize=font_size_2)
ax4.set_xticks(range(0,100,20))
ax4.plot(mu_std, label=r'$\textbf{s}$', color='m')
ax4.plot(l_std, label=r'$\textbf{l}$', color='c')
# ax4.ticklabel_format(axis="y", style="sci", scilimits=(0,00))
ax4.legend(prop={'size': font_size_3},frameon=False)
# plt.subplots_adjust(left=0.13, bottom=0.22, right=0.99, top=0.87, wspace=0.5, hspace=0.7)
plt.subplots_adjust(left=0.13, bottom=0.31, right=.99, top=0.86, wspace=0.4, hspace=0.7)
# plt.show(block=False)
fig2.savefig("param_var.pdf")

font_size_1 = 12*2
font_size_2 = 14*2
font_size_3 = 12*2
plt.rcParams.update({'font.size': font_size_1})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# plt.rcParams["figure.figsize"] = (12,4.5)
plt.rcParams["figure.figsize"] = (6,4.5)
plt.rcParams.update({'font.size': font_size_2})
rc('axes', linewidth=2)
fig3 = plt.figure()
plt.axis('off')
ax5 = fig3.add_subplot(1, 1, 1, projection='3d')
ax5.set_xlabel(r'$s_1$',fontsize=font_size_2,labelpad=1)
ax5.set_ylabel(r'$s_2$',fontsize=font_size_2,labelpad=1)
ax5.set_zlabel(r'$s_3$',fontsize=font_size_2,labelpad=1)
ax5.set_title(r'\textbf{(g)}',position=(0.2,.94), fontsize=font_size_2)
last_ep = peg_rnd_50_data[6]
success_mat = np.zeros(sample_num)
# selected_samplex = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
selected_samplex = [0,2, 8, 10, 11]
for s in selected_samplex:
    sample = last_ep[s]
    pos = sample['observations'][:,:3]
    min_pos = np.min(np.absolute(pos), axis=0)
    success_mat[s] = np.linalg.norm(min_pos) < SUCCESS_DIST
    if s==0:
        ax5.plot3D(pos[:, 0], pos[:, 1], pos[:, 2], color='b', linewidth=3,label='Trajectory')
    else:
        ax5.plot3D(pos[:, 0], pos[:, 1], pos[:, 2], color='b', linewidth=3)
# ax5.scatter(0,0,0,color='r',marker='o',s=50, label='Goal')
ax5.scatter(0,0,0,color='r',marker='o',s=50)
ax5.set_yticklabels([])
ax5.set_xticklabels([])
ax5.set_zticklabels([])
# plt.legend(loc='upper left', bbox_to_anchor=(.4, 0.15),frameon=False,ncol=2,prop={'size': font_size_3},)
plt.legend(loc='upper left', bbox_to_anchor=(0.12, 0.17),frameon=False, prop={'size': font_size_2},)
plt.subplots_adjust(left=0.0, bottom=0.04, right=1., top=1., wspace=0.0, hspace=0.0)
print('last_ep_success_rate',success_mat)

fig4 = plt.figure()
plt.axis('off')
ax6 = fig4.add_subplot(1, 1, 1, projection='3d')
ax6.set_xlabel(r'$s_1$',fontsize=font_size_2,labelpad=1)
ax6.set_ylabel(r'$s_2$',fontsize=font_size_2,labelpad=1)
ax6.set_zlabel(r'$s_3$',fontsize=font_size_2,labelpad=1)
ax6.set_title(r'\textbf{(f)}',position=(0.2,.94), fontsize=font_size_2)
first_ep = peg_rnd_0_data[9]
success_mat = np.zeros(sample_num)
# selected_samplex = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
selected_samplex = [2, 4, 6, 11, 12]
for s in selected_samplex:
    sample = first_ep[s]
    pos = sample['observations'][:,:3]
    min_pos = np.min(np.absolute(pos), axis=0)
    success_mat[s] = np.linalg.norm(min_pos) < SUCCESS_DIST
    if s == 2:
        # ax6.plot3D(pos[:, 0], pos[:, 1], pos[:, 2], color='b', linewidth=3,label='Trajectory')
        ax6.plot3D(pos[:, 0], pos[:, 1], pos[:, 2], color='b', linewidth=3)
    else:
        ax6.plot3D(pos[:, 0], pos[:, 1], pos[:, 2], color='b', linewidth=3)
ax6.scatter(0,0,0,color='r',marker='o',s=50, label='Goal')
ax6.set_yticklabels([])
ax6.set_xticklabels([])
ax6.set_zticklabels([])
print('first_ep_success_rate',success_mat)
# plt.legend(loc='upper left', bbox_to_anchor=(.4, 0.15),frameon=False,ncol=2,prop={'size': font_size_3},)
plt.legend(loc='upper left', bbox_to_anchor=(0.5, 0.17),frameon=False,prop={'size': font_size_2},)
plt.subplots_adjust(left=0.0, bottom=0.04, right=1., top=1., wspace=0.0, hspace=0.0)
# fig3.savefig("yumi_mjc_rand_init.pdf")
plt.show(block=True)
