import numpy as np
from garage.misc import tensor_utils

def cart_rwd_shape_1(d, v=1, w=1):

    alpha = 1e-5
    d_sq = d.dot(d)
    r = w*d_sq + v*np.log(d_sq + alpha) - v*np.log(alpha)
    assert (r >= 0)
    return r

def cart_rwd_func_1(x, f, terminal=False):
    '''
    This is for a regulation type problem, so x needs to go to zero.
    Magnitude of f has to be small
    :param x:
    :param f:
    :param g:
    :return:
    '''
    assert(x.shape==(12,))
    assert(f.shape==(6,))
    LIN_SCALE = 1
    ROT_SCALE = 1e-1
    POS_SCALE = 1
    VEL_SCALE = 1e-1
    STATE_SCALE = 1
    ACTION_SCALE = 1e-2
    v = 1
    w = 1
    TERMINAL_STATE_SCALE = 20.


    state_lin_pos_w = STATE_SCALE * LIN_SCALE * POS_SCALE
    state_rot_pos_w = STATE_SCALE * ROT_SCALE * POS_SCALE
    state_lin_vel_w = STATE_SCALE * LIN_SCALE * VEL_SCALE
    state_rot_vel_w = STATE_SCALE * ROT_SCALE * VEL_SCALE
    action_w = ACTION_SCALE

    x_lin_pos = x[:3]
    x_rot_pos = x[3:6]
    x_lin_vel = x[6:9]
    x_rot_vel = x[9:12]

    dx_lin_pos = cart_rwd_shape_1(x_lin_pos, v=v, w=w)
    dx_rot_pos = cart_rwd_shape_1(x_rot_pos, v=v, w=w)
    dx_lin_vel = x_lin_vel.dot(x_lin_vel)
    dx_rot_vel = x_rot_vel.dot(x_rot_vel)
    du = f.dot(f)

    reward_state_lin_pos = -state_lin_pos_w*dx_lin_pos
    reward_state_rot_pos = -state_rot_pos_w*dx_rot_pos
    if terminal:
        reward_state_lin_pos = TERMINAL_STATE_SCALE * reward_state_lin_pos
        reward_state_rot_pos = TERMINAL_STATE_SCALE * reward_state_rot_pos
    reward_state_lin_vel = -state_lin_vel_w*dx_lin_vel
    reward_state_rot_vel = -state_rot_vel_w*dx_rot_vel

    reward_state = reward_state_lin_pos + reward_state_rot_pos + reward_state_lin_vel + reward_state_rot_vel
    reward_action = -action_w*du
    reward = reward_state + reward_action
    rewards = np.array([reward_state_lin_pos, reward_state_rot_pos, reward_state_lin_vel, reward_state_rot_vel, reward_action])

    return reward, rewards

def process_cart_path_rwd(path, kin_obj, discount):


    Q_Qdots = path['observations']
    X_Xdots = kin_obj.get_cart_error_frame_list(Q_Qdots)
    N = Q_Qdots.shape[0]
    path['observations'] = X_Xdots
    Fs = path['agent_infos']['mean']
    Trqs = path['actions']
    path['actions'] = Fs
    path['agent_infos']['mean'] = Trqs
    Xs = X_Xdots[:,:12]
    Rxs = np.zeros((N,4))
    Rus = np.zeros(N)
    Rs = np.zeros(N)
    for i in range(N):
        x = Xs[i]
        f = Fs[i]
        r, rs = cart_rwd_func_1(x, f, terminal=(i==(N-1)))
        Rs[i] = r
        Rus[i] = rs[4]
        Rxs[i] = rs[:4]
    path['rewards'] = Rs
    path['env_infos']['reward_dist'] = Rxs
    path['env_infos']['reward_ctrl'] = Rus
    path['returns'] = tensor_utils.discount_cumsum(path['rewards'], discount)




