import numpy as np
from garage.misc import tensor_utils

def cart_rwd_func_1(x, f):
    '''
    This is for a regulation type problem, so x needs to go to zero.
    Magnitude of f has to be small
    :param x:
    :param f:
    :param g:
    :return:
    '''
    assert(x.shape==(6,))
    assert(f.shape==(6,))
    LIN_SCALE = 1
    ROT_SCALE = 1e-1
    STATE_SCALE = 1
    ACTION_SCALE = 1e-4
    SCALE_MAT = np.block([
                        [LIN_SCALE*np.eye(3), np.zeros((3,3))],
                        [np.zeros((3,3)), ROT_SCALE * np.eye(3)]
                        ])

    POS_FACTOR = STATE_SCALE * SCALE_MAT
    ACTION_FACTOR = ACTION_SCALE * SCALE_MAT


    reward_state = -x.T.dot(POS_FACTOR.dot(x))
    reward_action = -f.T.dot(ACTION_FACTOR.dot(f))
    reward = reward_state + reward_action

    return reward, reward_state, reward_action

def process_cart_path_rwd(path, kin_obj, discount):
    Q_Qdots = path['observations']
    X_Xdots = kin_obj.get_cart_error_frame_list(Q_Qdots)
    N = Q_Qdots.shape[0]
    path['observations'] = X_Xdots
    Fs = path['agent_infos']['mean']
    Trqs = path['actions']
    path['actions'] = Fs
    path['agent_infos']['mean'] = Trqs
    Ps = X_Xdots[:,:6]
    Rds = np.zeros(N)
    Rcs = np.zeros(N)
    Rs = np.zeros(N)
    for i in range(N):
        x = Ps[i]
        f = Fs[i]
        R, Rd, Rc = cart_rwd_func_1(x, f)
        Rs[i] = R
        Rds[i] = Rd
        Rcs[i] = Rc
    path['rewards'] = Rs
    path['env_infos']['reward_dist'] = Rds
    path['env_infos']['reward_ctrl'] = Rcs
    path['returns'] = tensor_utils.discount_cumsum(path['rewards'], discount)




