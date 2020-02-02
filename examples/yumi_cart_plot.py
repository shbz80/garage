import pickle
import matplotlib.pyplot as plt
import numpy as np
from gym.envs.mujoco.yumipeg import GOAL
from YumiKinematics import YumiKinematics

base_np_filename = '/home/shahbaz/Software/garage/examples/np/data/local/exp'
# exp_name ='test'
exp_name ='test_yumi_cart'
filename = base_np_filename + '/' + exp_name + '/' + 'exp_log.pkl'
infile = open(filename, 'rb')
exp_log = pickle.load(infile)
infile.close()

epoch_num = len(exp_log)
sample_num = len(exp_log[0])
T = exp_log[0][0]['observations'].shape[0]
tm = range(T)

yumikinparams = {}
yumikinparams['urdf'] = '/home/shahbaz/Software/mjc_models/yumi_ABB_left.urdf'
yumikinparams['base_link'] = 'world'
# yumikinparams['end_link'] = 'gripper_l_base'
yumikinparams['end_link'] = 'left_contact_point'
yumikinparams['euler_string'] = 'sxyz'
yumikinparams['goal'] = GOAL
yumiKin = YumiKinematics(yumikinparams)

SUCCESS_DIST = 0.05
SUCCESS_DIST_VEC = np.array([0.05, 0.01])
plot_skip = 5
plot_traj = True

for ep in range(epoch_num):
    epoch = exp_log[ep]
    obs0 = epoch[0]['observations']
    act0 = epoch[0]['actions']
    rwd_s0 = epoch[0]['env_infos']['reward_dist']
    rwd_a0 = epoch[0]['env_infos']['reward_ctrl']
    pos = obs0[:,:6].reshape(T,1,6)
    vel = obs0[:,6:].reshape(T,1,6)
    act = act0.reshape(T,1,6)
    rwd_s = rwd_s0.reshape(T,1)
    rwd_a = rwd_a0.reshape(T,1)


    cum_rwd_s_epoch = 0
    cum_rwd_a_epoch = 0
    # cum_rwd_t_epoch = 0
    for sp in range(sample_num):
        sample = epoch[sp]
        p = sample['observations'][:,:6].reshape(T,1,6)
        v = sample['observations'][:, 6:].reshape(T, 1, 6)
        a = sample['actions'].reshape(T, 1, 6)
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
        ax = fig.add_subplot(4, 6, 1)
        ax.set_title(r'$x$')
        ax.plot(tm, pos[:, :, 0], color='g')
        ax = fig.add_subplot(4, 6, 2)
        ax.set_title(r'$y$')
        ax.plot(tm, pos[:, :, 1], color='g')
        ax = fig.add_subplot(4, 6, 3)
        ax.set_title(r'$z$')
        ax.plot(tm, pos[:, :, 2], color='g')
        ax = fig.add_subplot(4, 6, 4)
        ax.set_title(r'$\alpha$')
        ax.plot(tm, pos[:, :, 3], color='g')
        ax = fig.add_subplot(4, 6, 5)
        ax.set_title(r'$\beta$')
        ax.plot(tm, pos[:, :, 4], color='g')
        ax = fig.add_subplot(4, 6, 6)
        ax.set_title(r'$\gamma$')
        ax.plot(tm, pos[:, :, 5], color='g')
        # plot Cartesian velocities
        ax = fig.add_subplot(4, 6, 7)
        ax.set_title(r'$\dot{x}$')
        ax.plot(tm, vel[:, :, 0], color='b')
        ax = fig.add_subplot(4, 6, 8)
        ax.set_title(r'$\dot{y}$')
        ax.plot(tm, vel[:, :, 1], color='b')
        ax = fig.add_subplot(4, 6, 9)
        ax.set_title(r'$\dot{z}$')
        ax.plot(tm, vel[:, :, 2], color='b')
        ax = fig.add_subplot(4, 6, 10)
        ax.set_title(r'$\dot{\alpha}$')
        ax.plot(tm, vel[:, :, 3], color='b')
        ax = fig.add_subplot(4, 6, 11)
        ax.set_title(r'$\dot{\beta}$')
        ax.plot(tm, vel[:, :, 4], color='b')
        ax = fig.add_subplot(4, 6, 12)
        ax.set_title(r'$\dot{\gamma}$')
        ax.plot(tm, vel[:, :, 5], color='b')
        # plot Cartesian forces
        ax = fig.add_subplot(4, 6, 13)
        ax.set_title(r'$\dot{x}$')
        ax.plot(tm, act[:, :, 0], color='r')
        ax = fig.add_subplot(4, 6, 14)
        ax.set_title(r'$\dot{y}$')
        ax.plot(tm, act[:, :, 1], color='r')
        ax = fig.add_subplot(4, 6, 15)
        ax.set_title(r'$\dot{z}$')
        ax.plot(tm, act[:, :, 2], color='r')
        ax = fig.add_subplot(4, 6, 16)
        ax.set_title(r'$\dot{\alpha}$')
        ax.plot(tm, act[:, :, 3], color='r')
        ax = fig.add_subplot(4, 6, 17)
        ax.set_title(r'$\dot{\beta}$')
        ax.plot(tm, act[:, :, 4], color='r')
        ax = fig.add_subplot(4, 6, 18)
        ax.set_title(r'$\dot{\gamma}$')
        ax.plot(tm, act[:, :, 5], color='r')
        # plot reward state
        ax = fig.add_subplot(4, 6, 19)
        ax.set_title(r'$r_s$')
        ax.plot(tm, rwd_s, color='m')
        # plot reward action
        ax = fig.add_subplot(4, 6, 20)
        ax.set_title(r'$r_a$')
        ax.plot(tm, rwd_a, color='c')
        # plot reward total
        ax = fig.add_subplot(4, 6, 21)
        ax.set_title(r'$r$')
        ax.plot(tm, rwd_s+rwd_a, color='c')

rewards_disc_rtn = np.zeros(epoch_num)
rewards_undisc_rwd = np.zeros(epoch_num)
success_mat = np.zeros((epoch_num, sample_num))
for ep in range(epoch_num):
    epoch = exp_log[ep]
    rewards_disc_rtn[ep] = np.mean([epoch[s]['returns'][0] for s in range(sample_num)])
    rewards_undisc_rwd[ep] = np.mean([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
    # for s in range(sample_num):
    #     success_mat[ep, s] = np.linalg.norm(epoch[s]['observations'][-1,:7] - GOAL) < SUCCESS_DIST

# success_stat = np.sum(success_mat, axis=1)*(100/sample_num)

fig = plt.figure()
plt.axis('off')
ax = fig.add_subplot(1, 2, 1)
ax.set_title('Progress')
ax.set_xlabel('Epoch')
ax.plot(rewards_undisc_rwd, label='undisc. reward')
ax.legend()
# ax = fig.add_subplot(1, 2, 2)
# ax.set_ylabel('Succes rate')
# ax.set_xlabel('Epoch')
# ax.plot(success_stat)
plt.show()
