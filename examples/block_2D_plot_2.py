import pickle
import matplotlib.pyplot as plt
import numpy as np
from gym.envs.mujoco.block2D import GOAL
# from glob import glob
base_np_filename = '/home/shahbaz/Software/garage/examples/np/data/local'
# exp_basename = 'mod_cem_block_KH_20e-2_'
# exp_basename = 'mod_cem_block_KH_10e-2_'
exp_basename = 'mod-cem-block-KH-20e-2-ns-20'
file_num = 10


fig_progress = plt.figure()
plt.axis('off')
ax_progress = fig_progress.add_subplot(1, 2, 1)
ax_progress.set_title('Progress')
ax_progress.set_xlabel('Epoch')
ax_success = fig_progress.add_subplot(1, 2, 2)
ax_success.set_ylabel('Succes rate')
ax_success.set_xlabel('Epoch')
for i in range(1,file_num+1):
    filename = base_np_filename + '/' + exp_basename + '/' + str(i) + '/' + 'exp_log.pkl'
    infile = open(filename, 'rb')
    exp_log = pickle.load(infile)
    infile.close()

    epoch_num = len(exp_log)
    sample_num = len(exp_log[0])
    T = exp_log[0][0]['observations'].shape[0]
    tm = range(T)
    # GOAL = np.array([0.4, -0.1])
    SUCCESS_DIST = 0.05
    SUCCESS_DIST_VEC = np.array([0.05, 0.01])
    plot_skip = 5
    plot_traj = False

    for ep in range(epoch_num):
        if ((ep==0) or (not ((ep+1) % plot_skip))) and plot_traj:
            epoch = exp_log[ep]
            obs0 = epoch[0]['observations']
            act0 = epoch[0]['actions']
            rwd_s0 = epoch[0]['env_infos']['reward_dist']
            rwd_a0 = epoch[0]['env_infos']['reward_ctrl']
            pos = obs0[:,:2].reshape(T,1,2)
            vel = obs0[:,2:4].reshape(T,1,2)
            act = act0.reshape(T,1,2)
            rwd_s = rwd_s0.reshape(T,1)
            rwd_a = rwd_a0.reshape(T,1)


            cum_rwd_s_epoch = 0
            cum_rwd_a_epoch = 0
            # cum_rwd_t_epoch = 0
            for sp in range(sample_num):
                sample = epoch[sp]
                p = sample['observations'][:,:2].reshape(T,1,2)
                v = sample['observations'][:, 2:4].reshape(T, 1, 2)
                a = sample['actions'].reshape(T, 1, 2)
                rs = sample['env_infos']['reward_dist'].reshape(T, 1)
                cum_rwd_s_epoch = cum_rwd_s_epoch + np.sum(rs.reshape(-1))
                ra = sample['env_infos']['reward_ctrl'].reshape(T, 1)
                cum_rwd_a_epoch = cum_rwd_a_epoch + np.sum(ra.reshape(-1))
                pos = np.concatenate((pos,p), axis=1)
                vel = np.concatenate((vel, v), axis=1)
                act = np.concatenate((act, a), axis=1)
                rwd_s = np.concatenate((rwd_s, rs), axis=1)
                rwd_a = np.concatenate((rwd_a, ra), axis=1)

            fig_rwd = plt.figure()
            plt.title('Epoch '+str(ep))
            plt.axis('off')
            ax = fig_rwd.add_subplot(2, 4, 1)
            ax.set_title('s1')
            ax.plot(tm, pos[:, :, 0], color='g')
            ax = fig_rwd.add_subplot(2, 4, 2)
            ax.set_title('s2')
            ax.plot(tm, pos[:, :, 1], color='g')
            ax = fig_rwd.add_subplot(2, 4, 3)
            ax.set_title('sdot1')
            ax.plot(tm, vel[:, :, 0], color='b')
            ax = fig_rwd.add_subplot(2, 4, 4)
            ax.set_title('sdot2')
            ax.plot(tm, vel[:, :, 1], color='b')
            ax = fig_rwd.add_subplot(2, 4, 5)
            ax.set_title('a1')
            ax.plot(tm, act[:, :, 0], color='r')
            ax = fig_rwd.add_subplot(2, 4, 6)
            ax.set_title('a2')
            ax.plot(tm, act[:, :, 1], color='r')
            ax = fig_rwd.add_subplot(2, 4, 7)
            ax.set_title('rs')
            ax.plot(tm, rwd_s, color='m')
            ax = fig_rwd.add_subplot(2, 4, 8)
            ax.set_title('ra')
            ax.plot(tm, rwd_a, color='c')

    rewards_disc_rtn = np.zeros(epoch_num)
    rewards_undisc_rwd = np.zeros(epoch_num)
    success_mat = np.zeros((epoch_num, sample_num))
    for ep in range(epoch_num):
        epoch = exp_log[ep]
        rewards_disc_rtn[ep] = np.mean([epoch[s]['returns'][0] for s in range(sample_num)])
        rewards_undisc_rwd[ep] = np.mean([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
        for s in range(sample_num):
            x_s = np.abs(epoch[s]['observations'][:, 0] - GOAL[0])<SUCCESS_DIST_VEC[0]
            y_s = np.abs(epoch[s]['observations'][:, 1] - GOAL[1]) < SUCCESS_DIST_VEC[1]
            xy_s = np.concatenate((x_s.reshape(-1,1), y_s.reshape(-1,1)), axis=1)
            success_mat[ep, s] = np.any(np.any(xy_s, axis=1))

    success_stat = np.sum(success_mat, axis=1)*(100/sample_num)

    # fig = plt.figure()


    ax_progress.plot(rewards_undisc_rwd, label='undisc. reward')
    ax_success.plot(success_stat)

plt.show()
# ax.legend()

