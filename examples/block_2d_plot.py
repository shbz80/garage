import pickle
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from gym.envs.mujoco.block2D import GOAL
from utils import iMOGIC_energy_block_vec, iMOGIC_energy_blocks

# base_np_filename = '/home/shahbaz/Software/garage/examples/np/data/local'
base_np_filename = '/home/shahbaz/Software/garage/examples/torch/data/local'

exp_name = 'ppo_block_2d'
# prefix = 'exp'
prefix = 'v35'

# prefix = 'test'
# exp_name = '1'

filename = base_np_filename + '/' + prefix + '/' + exp_name + '/' + 'exp_log.pkl'
infile = open(filename, 'rb')
exp_log = pickle.load(infile)
infile.close()

# filename = base_np_filename + '/' + prefix + '/' + exp_name + '/' + 'exp_param.pkl'
# infile = open(filename, 'rb')
# exp_param = pickle.load(infile)
# infile.close()

epoch_num = len(exp_log)
sample_num = len(exp_log[0])
T = exp_log[0][0]['observations'].shape[0]
tm = range(T)
# GOAL = np.array([0.4, -0.1])
SUCCESS_DIST = 0.025
plot_skip = 20
plot_traj = True

for ep in range(epoch_num):
    if ((ep==0) or (not ((ep+1) % plot_skip))) and plot_traj:
        epoch = exp_log[ep]
        obs0 = epoch[0]['observations']
        act0 = epoch[0]['actions']
        rwd_s0 = epoch[0]['env_infos']['reward_dist']
        rwd_a0 = epoch[0]['env_infos']['reward_ctrl']
        # param_ep = exp_param[ep]
        # param0 = param_ep['epoc_params'][0]
        # K = len(param0[2])
        # eng0 = iMOGIC_energy_block_vec(obs0[:,:2]-GOAL, obs0[:,2:4], param0, K, M=2)
        pos = obs0[:,:2].reshape(T,1,2)
        vel = obs0[:,2:4].reshape(T,1,2)
        act = act0.reshape(T,1,2)
        rwd_s = rwd_s0.reshape(T,1)
        rwd_a = rwd_a0.reshape(T,1)
        # eng = eng0.reshape(T,1)


        cum_rwd_s_epoch = 0
        cum_rwd_a_epoch = 0
        # cum_rwd_t_epoch = 0
        for sp in range(1,sample_num):
            sample = epoch[sp]
            p = sample['observations'][:,:2].reshape(T,1,2)
            v = sample['observations'][:, 2:4].reshape(T, 1, 2)
            a = sample['actions'].reshape(T, 1, 2)
            rs = sample['env_infos']['reward_dist'].reshape(T, 1)
            cum_rwd_s_epoch = cum_rwd_s_epoch + np.sum(rs.reshape(-1))
            ra = sample['env_infos']['reward_ctrl'].reshape(T, 1)
            cum_rwd_a_epoch = cum_rwd_a_epoch + np.sum(ra.reshape(-1))
            # param_ep = exp_param[ep]
            # param_sp = param_ep['epoc_params'][sp]
            # K = len(param_sp[2])
            # e = iMOGIC_energy_block_vec(sample['observations'][:,:2]-GOAL, sample['observations'][:,2:4], param_sp, K, M=2).reshape(T,1)
            pos = np.concatenate((pos,p), axis=1)
            vel = np.concatenate((vel, v), axis=1)
            act = np.concatenate((act, a), axis=1)
            rwd_s = np.concatenate((rwd_s, rs), axis=1)
            rwd_a = np.concatenate((rwd_a, ra), axis=1)
            # eng = np.concatenate((eng, e), axis=1)

        fig = plt.figure()
        plt.title('Epoch '+str(ep))
        plt.axis('off')
        ax = fig.add_subplot(3, 4, 1)
        ax.set_title('s1')
        ax.plot(tm, pos[:, :, 0], color='g')
        ax = fig.add_subplot(3, 4, 2)
        ax.set_title('s2')
        ax.plot(tm, pos[:, :, 1], color='g')
        ax = fig.add_subplot(3, 4, 3)
        ax.set_title('sdot1')
        ax.plot(tm, vel[:, :, 0], color='b')
        ax = fig.add_subplot(3, 4, 4)
        ax.set_title('sdot2')
        ax.plot(tm, vel[:, :, 1], color='b')
        ax = fig.add_subplot(3, 4, 5)
        ax.set_title('a1')
        ax.plot(tm, act[:, :, 0], color='r')
        ax = fig.add_subplot(3, 4, 6)
        ax.set_title('a2')
        ax.plot(tm, act[:, :, 1], color='r')
        ax = fig.add_subplot(3, 4, 7)
        ax.set_title('rs')
        ax.plot(tm, rwd_s, color='m')
        ax = fig.add_subplot(3, 4, 8)
        ax.set_title('ra')
        ax.plot(tm, rwd_a, color='c')
        # ax = fig.add_subplot(3, 4, 9)
        # ax.set_title('energy')
        # ax.plot(tm, eng, color='r')

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
        pos_norm = np.linalg.norm(epoch[s]['observations'][:, :2] - GOAL, axis=1)
        success_mat[ep, s] = np.min(pos_norm)<SUCCESS_DIST

success_stat = np.sum(success_mat, axis=1)*(100/sample_num)

fig = plt.figure()
plt.axis('off')
ax = fig.add_subplot(1, 2, 1)
ax.set_title('Progress')
ax.set_xlabel('Epoch')
ax.plot(rewards_undisc_mean, label='undisc. reward')
ax.legend()
ax = fig.add_subplot(1, 2, 2)
ax.set_ylabel('Succes rate')
ax.set_xlabel('Epoch')
ax.plot(success_stat)

# plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1, hspace=0.1)

# epoch = exp_log[0]
# obs0 = epoch[0]['observations']
# pos0 = obs0[:,:2] - GOAL
# T = pos0.shape[0]
# vel0 = np.zeros((T,2))
# param_ep = exp_param[0]
# param0 = param_ep['epoc_params'][0]
# K = len(param0[2])
#
# U = 30
# V = 30
# s1 = np.linspace(-0.2, 0.2, U)
# s2 = np.linspace(-0.7, 0.1, V)
# S1, S2 = np.meshgrid(s1, s2)
# F = np.zeros((U,V))
# for i in range(U):
#     for j in range(V):
#         S = np.array([S1[i,j], S2[i,j]])
#         S_dot = np.zeros(2)
#         F[i,j] = iMOGIC_energy_blocks(S, S_dot, param0, K, M=2)
#
# v0 = iMOGIC_energy_block_vec(pos0, vel0, param0, K, M=2)
# fig = plt.figure()
# ax = fig.add_subplot(1, 2, 1, projection='3d')
# # ax.contour3D(S1, S2, F, 50, cmap='viridis')
# ax.plot_wireframe(S1, S2, F, alpha=.7,label='Energy function')
# # ax.plot_surface(S1, S2, F,
# #                 cmap='viridis', edgecolor='none',alpha=0.7)
# ax.plot3D(pos0[:,0], pos0[:,1], v0, color='g',linewidth=3,label='Traj sample')
# ax.set_xlabel('s1')
# ax.set_ylabel('s2')
# ax.set_zlabel('v')
# # ax.legend()
#
#
# final_sample = 0
# epoch = exp_log[-1]
# obsl = epoch[final_sample]['observations']
# posl = obsl[:,:2] - GOAL
# param_ep = exp_param[-1]
# paraml = param_ep['epoc_params'][final_sample]
# F = np.zeros((U,V))
# for i in range(U):
#     for j in range(V):
#         S = np.array([S1[i,j], S2[i,j]])
#         S_dot = np.zeros(2)
#         F[i,j] = iMOGIC_energy_blocks(S, S_dot, paraml, K, M=2)
# vl = iMOGIC_energy_block_vec(posl, vel0, paraml, K, M=2)
# ax = fig.add_subplot(1, 2, 2, projection='3d')
# ax.plot_wireframe(S1, S2, F, alpha=.7,label='Energy function')
# # ax.contour3D(S1, S2, F, 50, cmap='viridis')
# # ax.plot_surface(S1, S2, F,
# #                 cmap=cm.coolwarm, edgecolor='none',alpha=1,linewidth=2)
# ax.plot3D(posl[:,0], posl[:,1], vl, color='g', linewidth=3, label='Traj sample')
# ax.set_xlabel('s1')
# ax.set_ylabel('s2')
# ax.set_zlabel('v')
# ax.legend()
# plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
plt.show()


############################################
'''
pos3 all failed.
show k16 1 for progress
'''
