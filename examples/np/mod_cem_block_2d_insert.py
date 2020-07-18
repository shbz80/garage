#!/usr/bin/env python3
"""
New modified Cross Entropy Method for PD matrices

Here it runs Block2D-v1 environment with 100 epoches.

Results:

"""
import numpy as np
import os
from garage.experiment import run_experiment
from garage.np.algos import MOD_CEM_SSD_BLOCKS
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import BatchSampler
from garage.envs import GarageEnv
from garage.experiment import LocalRunner
from garage.np.policies import StableSpringDamperPolicy
from gym.envs.mujoco.block2D import GOAL
import pickle

base_np_filename = '/home/shahbaz/Software/garage/examples/np/data/local'
# prefix = 'blocks-initpos3-K16'
# prefix = 'blocks_random_init_pos'
prefix = 'blocks_random_init_pos2'
exp_name = 'itr4'
# exp_name = 'itr49'
# exp_name = 'itr30'
# exp_name = '1'
# prefix = 'test'
# exp_name = 'temp'
dir_name = base_np_filename + '/' + prefix + '/' + exp_name
filename = dir_name + '/' + 'hyperparam.pkl'

def run_task(snapshot_config, *_):
    """Train CEM with Block2D-v1 environment."""
    with LocalRunner(snapshot_config=snapshot_config) as runner:
        env = GarageEnv(env_name='Block2D-v1')
        # K=2 components
        T = 200
        # n_samples = 15
        n_samples = 10
        # n_epochs = 100
        n_epochs = 1
        entropy_const = 1e0
        entropy_step_v = 100
        temperature = .1
        elite = True
        v_scalar_init = 2
        # K = 16
        K = 8
        best_frac = 0.2
        init_mu_mu = np.zeros(2)
        init_cov_diag = np.ones(2)
        M_norm = 0.4  # a value between 0 and 1.
        s=10.
        mass = 2.
        d=2*np.sqrt(mass*s)
        SD_mat_init = {}
        SD_mat_init['M_init'] = M_norm
        SD_mat_init['D_s'] = d / M_norm
        SD_mat_init['S_s'] = s / M_norm
        SD_mat_init['v'] = 3.
        SD_mat_init['local_scale'] = 1.

        exp_params = {}
        exp_params['T'] = T
        exp_params['n_samples'] = n_samples
        exp_params['n_epochs'] = n_epochs
        exp_params['entropy_const'] = entropy_const
        exp_params['entropy_step_v'] = entropy_step_v
        exp_params['temperature'] = temperature
        exp_params['elite'] = elite
        exp_params['v_scalar_init'] = v_scalar_init
        exp_params['K'] = K
        exp_params['best_frac'] = best_frac
        exp_params['init_mu_mu'] = init_mu_mu
        exp_params['init_cov_diag'] = init_cov_diag
        exp_params['M_norm'] = M_norm
        exp_params['s'] = s
        exp_params['d'] = d
        exp_params['SD_mat_init'] = SD_mat_init

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(filename, 'wb') as log_file:
            pickle.dump(exp_params, log_file)

        policy = StableSpringDamperPolicy(
                                      env.spec,
                                      GOAL,
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

        algo = MOD_CEM_SSD_BLOCKS(env_spec=env.spec,
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
                           temperature = temperature,
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
    exp_name=exp_name,
    exp_prefix=prefix,
    log_dir=None,
)

