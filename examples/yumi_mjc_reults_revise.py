import pickle
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from gym.envs.mujoco.yumipeg import GOAL, GOAL_CART
from YumiKinematics import YumiKinematics
from matplotlib import rc

font_size_1 = 12
font_size_2 = 14
font_size_3 = 10
plt.rcParams.update({'font.size': font_size_1})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

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
SUCCESS_DIST = .01

# _100 = '/home/shahbaz/Software/garage/examples/np/data/local/peg-woinit-ro15-beta10-Stinit100-Srinit1-lowv/1/exp_log.pkl'
# _150 = '/home/shahbaz/Software/garage/examples/np/data/local/peg-woinit-ro15-beta10-Stinit150-Srinit1-lowv/3/exp_log.pkl'
# _200 = '/home/shahbaz/Software/garage/examples/np/data/local/peg-woinit-ro15-beta10-Stinit200-Srinit4/3/exp_log.pkl'
# _250 = '/home/shahbaz/Software/garage/examples/np/data/local/peg-woinit-ro15-beta10-Stinit250-Srinit1-lowv/3/exp_log.pkl'
#
# param_exp_list= [_250, _200, _150, _100]
#
# color_list = ['b', 'g', 'm', 'c']
# # legend_list = ['$W_{S^0}^{init}=250$', '$W_{S^0}^{init}=200$', '$W_{S^0}^{init}=150$','$W_{S^0}^{init}=100$']
# legend_list = ['$250I$', '$200I$', '$150I$','$100I$']
# plt.rcParams["figure.figsize"] = (6,1.35)
# fig1 = plt.figure()
# plt.axis('off')
# ax1 = fig1.add_subplot(1, 2, 1)
# ax1.set_title(r'\textbf{(c)}', position=(-0.3, 0.94), fontsize=font_size_2)
# ax1.set_xlabel(r'Iteration')
# ax1.set_ylabel(r'Reward')
# ax1.set_xticks(range(0, 50, 20))
# ax1.set_ylim(-1.0e4,-0.1e4)
# ax1.set_xticks(range(0, 100, 20))
# ax2 = fig1.add_subplot(1, 2, 2)
# ax2.set_ylabel(r'Success \%')
# ax2.set_xlabel('Iteration')
# ax2.set_title(r'\textbf{(d)}', position=(-0.25, 0.94), fontsize=font_size_2)
# ax2.set_xticks(range(0, 50, 20))
# ax2.set_yticks([0,50,100])
# epoch_num = 50
#
# for i in range(len(param_exp_list)):
#     file = param_exp_list[i]
#     infile = open(file, 'rb')
#     log_data = pickle.load(infile)
#     infile.close()
#
#     # epoch_num = len(log_data)
#     sample_num = len(log_data[0])
#     T = log_data[0][0]['observations'].shape[0]
#     tm = range(T)
#
#     rewards_undisc_mean = np.zeros(epoch_num)
#     rewards_undisc_std = np.zeros(epoch_num)
#     success_mat = np.zeros((epoch_num, sample_num))
#
#     for ep in range(epoch_num):
#         epoch = log_data[ep]
#         rewards_undisc_mean[ep] = np.mean([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
#         rewards_undisc_std[ep] = np.std([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
#         for s in range(sample_num):
#             # sample = np.min(np.absolute(epoch[s]['observations'][:,:3]), axis=0)
#             sample = np.min(np.linalg.norm(epoch[s]['observations'][:, :3], axis=1), axis=0)
#             # sample = epoch[s]['observations'][:, :6]
#             # success_mat[ep, s] = np.linalg.norm(sample) < SUCCESS_DIST
#             success_mat[ep, s] = sample < SUCCESS_DIST
#
#     success_stat = np.sum(success_mat, axis=1)*(100/sample_num)
#
#     interval = 5
#     idx = i+1
#
#     inds = range(0,epoch_num)[idx::interval]
#     # inds = [0] + list(inds)
#     # del inds[-1]
#     heights = rewards_undisc_mean[idx::interval]
#     yerr = rewards_undisc_std[idx::interval]
#     yerr[0] = 0
#     ax1.errorbar(inds, heights, yerr=yerr, label=legend_list[i], color=color_list[i])
#     ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,00))
#     # ax1.legend(prop={'size': font_size_3},frameon=False)
#     ax1.legend(loc='upper left', bbox_to_anchor=(0.18, 0.6), frameon=False, ncol=2, prop={'size': font_size_3})
#
#     width = 1
#     inds = range(0, epoch_num)[idx::interval]
#     # inds = [0] + list(inds)
#     # del inds[-1]
#     success = success_stat[idx::interval]
#     ax2.bar(inds, success,width, color=color_list[i])
#     # ax2.legend(prop={'size': font_size_3},frameon=False)
# plt.subplots_adjust(left=0.13, bottom=0.31, right=.99, top=0.86, wspace=0.4, hspace=0.7)
# fig1.savefig("yumi_mjc_param_variation.pdf")

# initpos1 = '/home/shahbaz/Software/garage/examples/np/data/local/peg-winit-full-imped/2/exp_log.pkl'
# initpos2 = '/home/shahbaz/Software/garage/examples/np/data/local/peg-woinit-ro15-beta10-Stinit200-Srinit4-initpos2/2/exp_log.pkl'
initpos3 = '/home/shahbaz/Software/garage/examples/np/data/local/peg-woinit-ro15-beta10-Stinit200-Srinit4-initpos3/2/exp_log.pkl'
#
#
# param_exp_list= [initpos1, initpos2, initpos3]
#
# color_list = ['b', 'g', 'm']
# legend_list = ['$s_0^1$', '$s_0^2$', '$s_0^3$']
# plt.rcParams["figure.figsize"] = (6,2)
# fig1 = plt.figure()
# plt.axis('off')
# ax1 = fig1.add_subplot(1, 2, 1)
# ax1.set_title(r'\textbf{(a)}', position=(-0.3, 0.97), fontsize=font_size_2)
# ax1.set_xlabel(r'Iteration')
# ax1.set_ylabel(r'Reward')
# ax1.set_xticks(range(0, 100, 20))
# ax2 = fig1.add_subplot(1, 2, 2)
# ax2.set_ylabel(r'Success rate')
# ax2.set_xlabel('Iteration')
# ax2.set_title(r'\textbf{(b)}', position=(-0.25, 0.97), fontsize=font_size_2)
# ax2.set_xticks(range(0, 100, 20))
# epoch_num = 50
#
# for i in range(len(param_exp_list)):
#     file = param_exp_list[i]
#     infile = open(file, 'rb')
#     log_data = pickle.load(infile)
#     infile.close()
#
#     # epoch_num = len(log_data)
#     sample_num = len(log_data[0])
#     T = log_data[0][0]['observations'].shape[0]
#     tm = range(T)
#
#     rewards_undisc_mean = np.zeros(epoch_num)
#     rewards_undisc_std = np.zeros(epoch_num)
#     success_mat = np.zeros((epoch_num, sample_num))
#
#     for ep in range(epoch_num):
#         epoch = log_data[ep]
#         rewards_undisc_mean[ep] = np.mean([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
#         rewards_undisc_std[ep] = np.std([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
#         for s in range(sample_num):
#             # sample = np.min(np.absolute(epoch[s]['observations'][:,:3]), axis=0)
#             sample = np.min(np.linalg.norm(epoch[s]['observations'][:, :3], axis=1), axis=0)
#             # sample = epoch[s]['observations'][:, :6]
#             # success_mat[ep, s] = np.linalg.norm(sample) < SUCCESS_DIST
#             success_mat[ep, s] = sample < SUCCESS_DIST
#
#     success_stat = np.sum(success_mat, axis=1)*(100/sample_num)
#
#     interval = 5
#     idx = i+1
#
#     inds = range(0,epoch_num)[idx::interval]
#     # inds = [0] + list(inds)
#     # del inds[-1]
#     heights = rewards_undisc_mean[idx::interval]
#     yerr = rewards_undisc_std[idx::interval]
#     yerr[0] = 0
#     ax1.errorbar(inds, heights, yerr=yerr, label=legend_list[i], color=color_list[i])
#     ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,00))
#     ax1.legend(prop={'size': font_size_3},frameon=False)
#
#     width = 1
#     inds = range(0, epoch_num)[idx::interval]
#     # inds = [0] + list(inds)
#     # del inds[-1]
#     success = success_stat[idx::interval]
#     ax2.bar(inds, success,width, color=color_list[i])
#     ax2.legend(prop={'size': font_size_3},frameon=False)
# plt.subplots_adjust(left=0.13, bottom=0.22, right=.99, top=0.87, wspace=0.4, hspace=0.7)
# fig1.savefig("yumi_mjc_inipos_variation.pdf")
#
# _15 = '/home/shahbaz/Software/vices/vices_garage/data/local/exp/ppo_yumi_vices_ro15_itr100_range0.5/exp_log.pkl'
# _30 = '/home/shahbaz/Software/vices/vices_garage/data/local/exp/ppo_yumi_vices_ro30_itr100_range0.5/exp_log.pkl'
# _60 = '/home/shahbaz/Software/vices/vices_garage/data/local/exp/ppo_yumi_vices_ro60_itr100_range0.5/exp_log.pkl'
#
#
# param_exp_list= [_15, _30, _60]
# ppo_ro15_idx_offset = 1
ppo_ro30_idx_offset = 2
# inds_offest = [ppo_ro15_idx_offset, ppo_ro30_idx_offset, 3]
#
# color_list = ['b', 'g', 'm']
# legend_list = ['$N_s=15$', '$N_s=30$', '$N_s=60$']
# plt.rcParams["figure.figsize"] = (6,2)
# fig1 = plt.figure()
# plt.axis('off')
# ax1 = fig1.add_subplot(1, 2, 1)
# ax1.set_title(r'\textbf{(a)}', position=(-0.3, 0.97), fontsize=font_size_2)
# ax1.set_xlabel(r'Iteration')
# ax1.set_ylabel(r'Reward')
# ax1.set_xticks(range(0, 100, 20))
# ax1.set_ylim(-1.23e4,-.3e4)
# ax2 = fig1.add_subplot(1, 2, 2)
# ax2.set_ylabel(r'Success rate')
# ax2.set_xlabel('Iteration')
# ax2.set_title(r'\textbf{(b)}', position=(-0.25, 0.97), fontsize=font_size_2)
# ax2.set_xticks(range(0, 100, 20))
#
# # epoch_num = 50
#
# for i in range(len(param_exp_list)):
#     file = param_exp_list[i]
#     infile = open(file, 'rb')
#     log_data = pickle.load(infile)
#     infile.close()
#
#     epoch_num = len(log_data)
#     sample_num = len(log_data[0])
#     T = log_data[0][0]['observations'].shape[0]
#     tm = range(T)
#
#     rewards_undisc_mean = np.zeros(epoch_num)
#     rewards_undisc_std = np.zeros(epoch_num)
#     success_mat = np.zeros((epoch_num, sample_num))
#
#     for ep in range(epoch_num):
#         epoch = log_data[ep]
#         rewards_undisc_mean[ep] = np.mean([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
#         rewards_undisc_std[ep] = np.std([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
#         for s in range(sample_num):
#             # sample = np.min(np.absolute(epoch[s]['observations'][:,:3]), axis=0)
#             sample = np.min(np.linalg.norm(epoch[s]['observations'][:, :3]-GOAL_CART[:3], axis=1), axis=0)
#             # sample = epoch[s]['observations'][:, :6]
#             # success_mat[ep, s] = np.linalg.norm(sample) < SUCCESS_DIST
#             success_mat[ep, s] = sample < SUCCESS_DIST
#
#     success_stat = np.sum(success_mat, axis=1)*(100/sample_num)
#
#     interval = 10
#     idx = i+1
#
#     inds = range(0,epoch_num)[idx::interval]
#     # inds = [0] + list(inds)
#     # del inds[-1]
#     heights = rewards_undisc_mean[inds_offest[i]::interval]
#     yerr = rewards_undisc_std[inds_offest[i]::interval]
#     yerr[0] = 0
#     ax1.errorbar(inds, heights, yerr=yerr, label=legend_list[i], color=color_list[i])
#     ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,00))
#     ax1.legend(prop={'size': font_size_3},frameon=False)
#
#     width = 2
#     inds = range(0, epoch_num)[idx+i::interval]
#     # inds = [0] + list(inds)
#     # del inds[-1]
#     success = success_stat[inds_offest[i]::interval]
#     ax2.bar(inds, success,width, color=color_list[i])
#     ax2.legend(prop={'size': font_size_3},frameon=False)
# plt.subplots_adjust(left=0.13, bottom=0.22, right=.99, top=0.87, wspace=0.4, hspace=0.7)
# fig1.savefig("yumi_mjc_ppo_tuning_Ns.pdf")
#
# _05 = '/home/shahbaz/Software/vices/vices_garage/data/local/exp/ppo_yumi_vices_ro15_itr100_range0.5/exp_log.pkl'
# _01 = '/home/shahbaz/Software/vices/vices_garage/data/local/exp/ppo_yumi_vices_ro15_itr100_range0.1/exp_log.pkl'
# _10 = '/home/shahbaz/Software/vices/vices_garage/data/local/exp/ppo_yumi_vices_ro15_itr100_range1/exp_log.pkl'
#
#
# param_exp_list= [_01, _05, _10]
# # param_exp_list= [_10]
# inds_offest = [ppo_ro15_idx_offset, 2, 3]
# color_list = ['b', 'g', 'm']
# legend_list = ['$a_r=0.1$', '$a_r=0.5$', '$a_r=1.0$']
# plt.rcParams["figure.figsize"] = (6,2)
# fig1 = plt.figure()
# plt.axis('off')
# ax1 = fig1.add_subplot(1, 2, 1)
# ax1.set_title(r'\textbf{(a)}', position=(-0.3, 0.97), fontsize=font_size_2)
# ax1.set_xlabel(r'Iteration')
# ax1.set_ylabel(r'Reward')
# ax1.set_xticks(range(0, 100, 20))
# ax1.set_ylim(-4.e4,-0.5e4)
# ax2 = fig1.add_subplot(1, 2, 2)
# ax2.set_ylabel(r'Success rate')
# ax2.set_xlabel('Iteration')
# ax2.set_title(r'\textbf{(b)}', position=(-0.25, 0.97), fontsize=font_size_2)
# ax2.set_xticks(range(0, 100, 20))
# ax2.set_ylim(0,100)
# # epoch_num = 50
#
# for i in range(len(param_exp_list)):
#     file = param_exp_list[i]
#     infile = open(file, 'rb')
#     log_data = pickle.load(infile)
#     infile.close()
#
#     epoch_num = len(log_data)
#     sample_num = len(log_data[0])
#     T = log_data[0][0]['observations'].shape[0]
#     tm = range(T)
#
#     rewards_undisc_mean = np.zeros(epoch_num)
#     rewards_undisc_std = np.zeros(epoch_num)
#     success_mat = np.zeros((epoch_num, sample_num))
#
#     for ep in range(epoch_num):
#         epoch = log_data[ep]
#         rewards_undisc_mean[ep] = np.mean([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
#         rewards_undisc_std[ep] = np.std([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
#         for s in range(sample_num):
#             # sample = np.min(np.absolute(epoch[s]['observations'][:,:3]), axis=0)
#             sample = np.min(np.linalg.norm(epoch[s]['observations'][:, :3]-GOAL_CART[:3], axis=1), axis=0)
#             # sample = epoch[s]['observations'][:, :6]
#             # success_mat[ep, s] = np.linalg.norm(sample) < SUCCESS_DIST
#             success_mat[ep, s] = sample < SUCCESS_DIST
#
#     success_stat = np.sum(success_mat, axis=1)*(100/sample_num)
#
#     interval = 10
#     idx = i+1+3
#
#     inds = range(0,epoch_num)[idx::interval]
#     # inds = [0] + list(inds)
#     # del inds[-1]
#     heights = rewards_undisc_mean[idx::interval]
#     yerr = rewards_undisc_std[idx::interval]
#     yerr[0] = 0
#     # ax1.errorbar(inds, heights, yerr=yerr, label=legend_list[i], color=color_list[i])
#     ax1.plot(inds, heights, label=legend_list[i], color=color_list[i])
#     ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,00))
#     ax1.legend(prop={'size': font_size_3},frameon=False)
#
#     width = 2
#     inds = range(0, epoch_num)[idx+i::interval]
#     # inds = [0] + list(inds)
#     # del inds[-1]
#     success = success_stat[inds_offest[i]::interval]
#     ax2.bar(inds, success,width, color=color_list[i])
#     ax2.legend(prop={'size': font_size_3},frameon=False)
# plt.subplots_adjust(left=0.13, bottom=0.22, right=.99, top=0.87, wspace=0.4, hspace=0.7)
# fig1.savefig("yumi_mjc_ppo_tuning_crange.pdf")
#
woinit = '/home/shahbaz/Software/garage/examples/np/data/local/peg-woinit-full-imped/1//exp_log.pkl'
winit = '/home/shahbaz/Software/garage/examples/np/data/local/peg-winit-full-imped/2/exp_log.pkl'
ppo_15 = '/home/shahbaz/Software/vices/vices_garage/data/local/exp/ppo_yumi_vices_ro15_itr100_range0.5/exp_log.pkl'
ppo_30 = '/home/shahbaz/Software/vices/vices_garage/data/local/exp/ppo_yumi_vices_ro30_itr100_range0.5/exp_log.pkl'



# param_exp_list= [woinit, winit, ppo_30]
# inds_offest = [1,3, ppo_ro30_idx_offset]
# color_list = ['b', 'g', 'm']
# legend_list = ['ours w/o init', 'ours with init', 'VICES']
# ns_list=[15,15,30]

param_exp_list= [winit, ppo_30]
inds_offest = [3, ppo_ro30_idx_offset]
color_list = ['b', 'g']
legend_list = ['ours', 'VICES']
ns_list=[15,30]

plt.rcParams["figure.figsize"] = (6,1.35)
fig1 = plt.figure()
plt.axis('off')
ax1 = fig1.add_subplot(1, 2, 1)
# ax1.set_title(r'\textbf{(a)}', position=(-0.3, 0.97), fontsize=font_size_2)
ax1.set_xlabel(r'Rollouts')
ax1.set_ylabel(r'Reward')
ax1.set_xticks(range(0, 3000, 600))
# ax1.set_ylim(-2.3e4,-0.1e4)
ax2 = fig1.add_subplot(1, 2, 2)
ax2.set_ylabel(r'Success \%')
ax2.set_xlabel('Rollouts')
# ax2.set_title(r'\textbf{(b)}', position=(-0.25, 0.97), fontsize=font_size_2)
ax2.set_xticks(range(0, 3000, 600))
ax2.set_ylim(0,105)
ax2.set_yticks([0,50,100])
# epoch_num = 50
ax1.set_title(r'\textbf{(a)}', position=(-0.3, 0.94), fontsize=font_size_2)
ax2.set_title(r'\textbf{(b)}', position=(-0.25, 0.94), fontsize=font_size_2)
for i in range(len(param_exp_list)):
    file = param_exp_list[i]
    infile = open(file, 'rb')
    log_data = pickle.load(infile)
    infile.close()

    epoch_num = len(log_data)
    if epoch_num>100:
        epoch_num = 100
    sample_num = len(log_data[0])
    T = log_data[0][0]['observations'].shape[0]
    tm = range(T)

    rewards_undisc_mean = np.zeros(epoch_num)
    rewards_undisc_std = np.zeros(epoch_num)
    success_mat = np.zeros((epoch_num, sample_num))

    for ep in range(epoch_num):
        epoch = log_data[ep]
        rewards_undisc_mean[ep] = np.mean([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
        rewards_undisc_std[ep] = np.std([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
        for s in range(sample_num):
            # sample = np.min(np.absolute(epoch[s]['observations'][:,:3]), axis=0)
            if file == woinit or file == winit:
                sample = np.min(np.linalg.norm(epoch[s]['observations'][:, :3], axis=1), axis=0)
            else:
                sample = np.min(np.linalg.norm(epoch[s]['observations'][:, :3] - GOAL_CART[:3], axis=1), axis=0)
            # sample = epoch[s]['observations'][:, :6]
            # success_mat[ep, s] = np.linalg.norm(sample) < SUCCESS_DIST
            success_mat[ep, s] = sample < SUCCESS_DIST

    success_stat = np.sum(success_mat, axis=1)*(100/sample_num)

    interval = 10
    idx = i

    # inds = range(0,epoch_num)[idx::interval]
    inds = np.array(range(0, epoch_num)[idx::interval])*ns_list[i]
    # inds = [0] + list(inds)
    # del inds[-1]
    heights = rewards_undisc_mean[idx::interval]
    yerr = rewards_undisc_std[idx::interval]
    yerr[0] = 0
    ax1.errorbar(inds, heights, yerr=yerr, label=legend_list[i], color=color_list[i])
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,00))
    # ax1.legend(loc='upper left', bbox_to_anchor=(0.37, 0.88), frameon=False, ncol=1, prop={'size': font_size_3})
    ax1.legend(loc='upper left', bbox_to_anchor=(0.45, 0.63), frameon=False, ncol=1, prop={'size': font_size_3})

    # if file is not winit:
    width = 2
    interval = interval*15//ns_list[i]
    inds = np.array(range(0, epoch_num)[idx*width::interval])*ns_list[i]
    # print(inds)
    # inds = [0] + list(inds)
    # del inds[-1]
    success = success_stat[inds_offest[i]::interval]
    # success = success_stat[inds]
    ax2.bar(inds, success,width*30, color=color_list[i])
    # ax2.legend(prop={'size': font_size_3},frameon=False)
    plt.show(block=False)
    None
# plt.subplots_adjust(left=0.13, bottom=0.22, right=.99, top=0.87, wspace=0.4, hspace=0.7)
plt.subplots_adjust(left=0.13, bottom=0.31, right=.99, top=0.86, wspace=0.4, hspace=0.7)
fig1.savefig("yumi_mjc_progress_with_vices.pdf")
#
plt.rcParams["figure.figsize"] = (2,2)
# VICES = '/home/shahbaz/Software/vices/vices_garage/data/local/exp/ppo_yumi_vices_ro15_itr100_range0.5/exp_log.pkl'
VICES = '/home/shahbaz/Software/vices/vices_garage/data/local/exp/ppo_yumi_vices_ro15_itr100_range0.5_initpos3/exp_log.pkl'
# ours = '/home/shahbaz/Software/garage/examples/np/data/local/peg-woinit-ro15-beta10-Stinit200-Srinit4/3/exp_log.pkl'

file = initpos3
# file = ours
infile = open(file, 'rb')
log_data = pickle.load(infile)
infile.close()
# epoch_num = len(log_data)
epoch_num = 10
sample_num = len(log_data[0])
T = log_data[0][0]['observations'].shape[0]
tm = range(T)
dist_ours = np.zeros((epoch_num, sample_num, T))
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(1, 2, 1)
last_dist_ours = np.zeros((epoch_num,sample_num))
for ep in range(epoch_num):
    epoch = log_data[ep]
    for s in range(sample_num):
        dist_ours[ep][s] = np.linalg.norm(epoch[s]['observations'][:, :3],axis=1).reshape(-1)
        last_dist_ours[ep][s] = np.linalg.norm(epoch[s]['observations'][:, :3], axis=1)[-1]
        # ax1.plot(dist_ours[ep][s])

dist_ours = dist_ours.reshape(-1)
last_dist_ours = last_dist_ours.reshape(-1)
file = VICES
infile = open(file, 'rb')
log_data = pickle.load(infile)
infile.close()
# epoch_num = len(log_data)
sample_num = len(log_data[0])
T = log_data[0][0]['observations'].shape[0]
tm = range(T)
dist_vices = np.zeros((epoch_num, sample_num, T))
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(1, 2, 1)
last_dist_vices = np.zeros((epoch_num,sample_num))
for ep in range(epoch_num):
    epoch = log_data[ep]
    for s in range(sample_num):
        dist_vices[ep][s] = np.linalg.norm(epoch[s]['observations'][:, :3] - GOAL_CART[:3],axis=1).reshape(-1)
        last_dist_vices[ep][s] = np.linalg.norm(epoch[s]['observations'][:, :3], axis=1)[-1]
        # ax1.plot(dist_vices[ep][s])
dist_vices = dist_vices.reshape(-1)
last_dist_vices = last_dist_vices.reshape(-1)
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.set_title(r'\textbf{(a)}', position=(-0.65, 0.86), fontsize=font_size_2)
# ax1.set_xlabel(r'Iteration')
# ax1.set_ylabel(r'Reward')
ax1.set_xticks([0.0,0.4,0.8])
# ax1.set_ylim(-1.7e4,-0.2e4)
# data = [dist_ours, dist_vices, last_dist_ours, last_dist_vices]
data = [last_dist_vices, last_dist_ours, dist_vices, dist_ours]
# ax1.boxplot(data, showfliers=False, whis=(0,100),vert=False)
bp = ax1.boxplot(data, patch_artist = False, showfliers=False, whis=(0,100),vert=False)
for median in bp['medians']:
    median.set(color ='blue',
               linewidth = 1)
# ax1.boxplot(data, showfliers=False)
# ax1.set_yticklabels(['Ours\n(all pos)', 'VICES\n(all pos)','Ours\n(final pos)', 'VICES\n(final pos)'])
ax1.set_yticklabels(['Final pos\n(VICES)','Final pos\n(ours)','All pos\n(VICES)','All pos\n(ours)'])
ax1.set_xlabel('m',labelpad=1)
plt.subplots_adjust(left=0.429, bottom=0.2, right=.99, top=0.98, wspace=0.5, hspace=0.7)
fig1.savefig("yumi_mjc_traj_spread.pdf")

# plt.show(block=False)
