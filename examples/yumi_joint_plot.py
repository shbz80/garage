import pickle
import matplotlib.pyplot as plt
import numpy as np
from gym.envs.mujoco.yumipeg import GOAL

base_np_filename = '/home/shahbaz/Software/garage/examples/np/data/local/exp'
exp_name ='test'
filename = base_np_filename + '/' + exp_name + '/' + 'exp_log.pkl'
infile = open(filename, 'rb')
exp_log = pickle.load(infile)
infile.close()

epoch_num = len(exp_log)
sample_num = len(exp_log[0])
T = exp_log[0][0]['observations'].shape[0]
tm = range(T)

SUCCESS_DIST = 0.01
SUCCESS_DIST_VEC = np.array([0.05, 0.01])
plot_skip = 5
plot_traj = True

for ep in range(epoch_num):
    epoch = exp_log[ep]
    obs0 = epoch[0]['observations']
    act0 = epoch[0]['actions']
    rwd_s0 = epoch[0]['env_infos']['reward_dist']
    rwd_a0 = epoch[0]['env_infos']['reward_ctrl']
    pos = obs0[:,:7].reshape(T,1,7)
    vel = obs0[:,7:].reshape(T,1,7)
    act = act0.reshape(T,1,7)
    rwd_s = rwd_s0.reshape(T,1)
    rwd_a = rwd_a0.reshape(T,1)


    cum_rwd_s_epoch = 0
    cum_rwd_a_epoch = 0
    # cum_rwd_t_epoch = 0
    for sp in range(sample_num):
        sample = epoch[sp]
        p = sample['observations'][:,:7].reshape(T,1,7)
        v = sample['observations'][:, 7:].reshape(T, 1, 7)
        a = sample['actions'].reshape(T, 1, 7)
        rs = sample['env_infos']['reward_dist'].reshape(T, 1)
        cum_rwd_s_epoch = cum_rwd_s_epoch + np.sum(rs.reshape(-1))
        ra = sample['env_infos']['reward_ctrl'].reshape(T, 1)
        cum_rwd_a_epoch = cum_rwd_a_epoch + np.sum(ra.reshape(-1))
        pos = np.concatenate((pos,p), axis=1)
        vel = np.concatenate((vel, v), axis=1)
        act = np.concatenate((act, a), axis=1)
        rwd_s = np.concatenate((rwd_s, rs), axis=1)
        rwd_a = np.concatenate((rwd_a, ra), axis=1)

    if ((ep == 0) or (not ((ep + 1) % plot_skip))) and plot_traj:
        fig = plt.figure()
        plt.title('Epoch '+str(ep))
        plt.axis('off')
        # plot Cartesian positions
        for i in range(7):
            ax = fig.add_subplot(4, 7, i+1)
            ax.set_title(r'$q_{%d}$' %(i+1))
            ax.plot(tm, pos[:, :, i], color='g')

            ax = fig.add_subplot(4, 7, 7+i+1)
            ax.set_title(r'$\dot{q}_{%d}$' %(i+1))
            ax.plot(tm, vel[:, :, i], color='b')

            ax = fig.add_subplot(4, 7, 7*2+i+1)
            ax.set_title(r'$u_{%d}$' %(i+1))
            ax.plot(tm, act[:, :, i], color='r')

        # plot reward state
        ax = fig.add_subplot(4, 7, 22)
        ax.set_title(r'$r_s$')
        ax.plot(tm, rwd_s, color='m')
        # plot reward action
        ax = fig.add_subplot(4, 7, 23)
        ax.set_title(r'$r_a$')
        ax.plot(tm, rwd_a, color='c')
        # plot reward total
        ax = fig.add_subplot(4, 7, 24)
        ax.set_title(r'$r$')
        ax.plot(tm, rwd_s+rwd_a, color='y')

rewards_disc_rtn = np.zeros(epoch_num)
rewards_undisc_rwd = np.zeros(epoch_num)
success_mat = np.zeros((epoch_num, sample_num))
for ep in range(epoch_num):
    epoch = exp_log[ep]
    rewards_disc_rtn[ep] = np.mean([epoch[s]['returns'][0] for s in range(sample_num)])
    rewards_undisc_rwd[ep] = np.mean([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
    for s in range(sample_num):
        success_mat[ep, s] = np.linalg.norm(epoch[s]['observations'][-1,:7] - GOAL) < SUCCESS_DIST

success_stat = np.sum(success_mat, axis=1)*(100/sample_num)

fig = plt.figure()
plt.axis('off')
ax = fig.add_subplot(1, 2, 1)
ax.set_title('Progress')
ax.set_xlabel('Epoch')
ax.plot(rewards_undisc_rwd, label='undisc. reward')
ax.legend()
ax = fig.add_subplot(1, 2, 2)
ax.set_ylabel('Succes rate')
ax.set_xlabel('Epoch')
ax.plot(success_stat)

plt.show()
