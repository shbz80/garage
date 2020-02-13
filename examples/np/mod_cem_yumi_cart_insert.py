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
from gym.envs.mujoco.yumipeg import GOAL


# GOAL = np.array([-1.63688, -1.22777, 1.28612, 0.446995, 2.21936, 1.57011, 0.47748])
def run_task(snapshot_config, *_):
    """Train CEM with Block2D-v1 environment."""
    with LocalRunner(snapshot_config=snapshot_config) as runner:
        env = GarageEnv(env_name='YumiPeg-v1')
        # K=7 components
        kin_params_yumi = {}
        kin_params_yumi['urdf'] = '/home/shahbaz/Software/mjc_models/yumi_ABB_left.urdf'
        kin_params_yumi['base_link'] = 'world'
        # kin_params_yumi['end_link'] = 'gripper_l_base'
        kin_params_yumi['end_link'] = 'left_contact_point'
        kin_params_yumi['euler_string'] = 'sxyz'
        kin_params_yumi['goal'] = GOAL

        policy = StableCartSpringDamperPolicy(
            env.spec,
            GOAL,
            kin_params_yumi,
            K=2,
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        # n_samples = 20
        n_samples = 15  # number of samples in an epoch in CEM

        # itr is the number of RL iterations consisting of a number of rollouts
        # for CEM when we use deterministic policy then we need only one RL rollout,
        # so in this case one itr consists of only one rollout.
        # number of rollouts is automatically determined from path_length and batch_size
        # we use path_length=100 and batch_size=100
        # In other RL algos one epoch consists of iteration, but in CEM one epoc corresponds
        # to one iteration of CEM that consists of n_samples rollouts.

        T = 40  # episode length

        algo = MOD_CEM_SSD(env_spec=env.spec,
                           policy=policy,
                           baseline=baseline,
                           # best_frac=0.05,
                           best_frac=0.2,
                           max_path_length=T,
                           n_samples=n_samples,
                           init_cov=1.,
                           init_pd_gain=2,
                           elite=True,
                           temperature=.1,
                           # entropy_const=1e1,
                           entropy_const=1e1,
                           entropy_step_v=100,
                           )
        # ***important change T in block2D.py (reward def) equal to max_path_length***
        runner.setup(algo, env, sampler_cls=BatchSampler)
        # NOTE: make sure that n_epoch_cycles == n_samples !
        # TODO: it is not clear why the above is required
        # runner.train(n_epochs=100, batch_size=1000, n_epoch_cycles=n_samples, plot=True, store_paths=True)
        runner.train(n_epochs=40, batch_size=T, n_epoch_cycles=n_samples, plot=False, store_paths=False)


# IMPORTANT: change the log directory in batch_polopt.py
run_experiment(
    run_task,
    snapshot_mode='last',
    # seed=1,
    plot=False,
    # exp_name='mod_cem_block_KH_10e-2_10',
    # exp_name='10',
    exp_name='test_yumi_cart',
    # exp_prefix='mod_cem_block_KH_20e-2_ns_20',
    exp_prefix='exp',
    log_dir=None,
)

