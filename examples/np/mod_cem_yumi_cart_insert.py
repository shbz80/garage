#!/usr/bin/env python3
"""
New modified Cross Entropy Method for PD matrices

Here it runs YumiPeg-v1 environment with 100 epoches.

Results:

"""
import numpy as np
from garage.experiment import run_experiment
from garage.np.algos import MOD_CEM_SSD
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import BatchSampler
from garage.envs import GarageEnv
from garage.experiment import LocalRunner
from garage.np.policies import StableCartSpringDamperPolicy
from gym.envs.mujoco.yumipeg import GOAL, INIT
from yumikin.YumiKinematics import YumiKinematics

def run_task(snapshot_config, *_):
    """Train CEM with Block2D-v1 environment."""
    with LocalRunner(snapshot_config=snapshot_config) as runner:
        env = GarageEnv(env_name='YumiPeg-v1')
        # K=7 components
        kin_params_yumi = {}
        kin_params_yumi['urdf'] = '/home/shahbaz/Software/yumi_kinematics/yumikin/models/yumi_ABB_left.urdf'
        kin_params_yumi['base_link'] = 'world'
        # kin_params_yumi['end_link'] = 'left_tool0'
        kin_params_yumi['end_link'] = 'left_contact_point'
        kin_params_yumi['euler_string'] = 'sxyz'
        kin_params_yumi['goal'] = GOAL

        T = 200  # episode length

        yumiKin = YumiKinematics(kin_params_yumi)
        x_d_i, _, _ = yumiKin.get_cart_error_frame_terms(INIT, np.zeros(7))
        delat_goal_cart_dist = x_d_i[:3]
        delat_goal_rot_dist = x_d_i[3:]
        assert (np.any(np.abs(delat_goal_rot_dist) < np.pi / 2))  # check if any rotation coordinate is more than pi/2
        r_cart_sq = np.square(np.linalg.norm(delat_goal_cart_dist))
        r_cart_comp_sq = r_cart_sq * np.ones(3)
        r_rot_sq = np.square(np.pi / 4)  # so that 2-sigma is pi/2
        r_rot_comp_sq = r_rot_sq * np.ones(3)
        init_mu_cov_diag = np.concatenate((r_cart_comp_sq, r_rot_comp_sq))
        # init_mu_cov_diag = np.ones(6)
        init_mu_mu = np.zeros(6)
        M_norm = 0.4  # a value between 0 and 1.
        # s_trans = 200.
        # s_rot = 4.
        s_trans = 200.
        s_rot = 4.
        S0_init = np.diag(np.array([s_trans, s_trans, s_trans, s_rot, s_rot, s_rot]))
        M_d_x = yumiKin.get_cart_intertia_d(INIT)
        M_d = np.diag(M_d_x)
        D_d = np.sqrt(np.multiply(M_d, np.diag(S0_init)))
        print('D_init:', D_d)
        d_trans = np.max(D_d[:3])
        d_rot = np.max(D_d[3:])
        # d_trans = 1
        # d_rot = 1
        SD_mat_init = {}
        SD_mat_init['M_init'] = M_norm
        SD_mat_init['D_trans_s'] = d_trans / M_norm
        SD_mat_init['D_rot_s'] = d_rot / M_norm
        SD_mat_init['S_trans_s'] = s_trans / M_norm
        SD_mat_init['S_rot_s'] = s_rot / M_norm
        SD_mat_init['v'] = 30.
        # SD_mat_init['v'] = 8.
        SD_mat_init['local_scale'] = 4.

        n_samples = 15  # number of samples in an epoch in CEM
        n_epochs = 50
        entropy_const = 1.0e1
        v_scalar_init = 20
        # v_scalar_init = 2
        K = 2
        best_frac = 0.2
        # best_frac = 0.1
        init_cov_diag = init_mu_cov_diag
        elite = True
        temperature = .1
        entropy_step_v = 100

        policy = StableCartSpringDamperPolicy(
            env.spec,
            GOAL,
            kin_params_yumi,
            T,
            K=K,
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        # itr is the number of RL iterations consisting of a number of rollouts
        # for CEM when we use deterministic policy then we need only one RL rollout,
        # so in this case one itr consists of only one rollout.
        # number of rollouts is automatically determined from path_length and batch_size
        # we use path_length=100 and batch_size=100
        # In other RL algos one epoch consists of iteration, but in CEM one epoc corresponds
        # to one iteration of CEM that consists of n_samples rollouts.



        algo = MOD_CEM_SSD(env_spec=env.spec,
                           policy=policy,
                           baseline=baseline,
                           best_frac=best_frac,
                           max_path_length=T,
                           n_samples=n_samples,
                           init_cov_diag=init_cov_diag,
                           SD_mat_init = SD_mat_init,
                           v_scalar_init=v_scalar_init,
                           mu_init=init_mu_mu,
                           elite=elite,
                           temperature=temperature,
                           entropy_const=entropy_const,
                           entropy_step_v=entropy_step_v,
                           )
        # ***important change T in block2D.py (reward def) equal to max_path_length***
        runner.setup(algo, env, sampler_cls=BatchSampler)
        # NOTE: make sure that n_epoch_cycles == n_samples !
        # TODO: it is not clear why the above is required
        # runner.train(n_epochs=100, batch_size=1000, n_epoch_cycles=n_samples, plot=True, store_paths=True)
        runner.train(n_epochs=n_epochs, batch_size=T, n_epoch_cycles=n_samples, plot=True, store_paths=False)


# IMPORTANT: change the log directory in batch_polopt.py
run_experiment(
    run_task,
    snapshot_mode='last',
    # seed=1,
    plot=True,
    # exp_name='mod_cem_block_KH_10e-2_10',
    # exp_name='10',
    exp_name='3',
    # exp_prefix='mod_cem_block_temp',
    # exp_prefix='peg_winit_partial_imped',
    # exp_prefix='peg_winit_full_imped',
    # exp_prefix='peg_random_init_pos_itr50',
    # exp_prefix='peg-woinit-ro15-beta10-Stinit200-Srinit4-initpos2',
    exp_prefix='temp',


    log_dir=None,
)

