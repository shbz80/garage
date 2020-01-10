#!/usr/bin/env python3
"""
New modified Cross Entropy Method for PD matrices

Here it runs Block2D-v1 environment with 100 epoches.

Results:

"""
import numpy as np
from garage.experiment import run_experiment
from garage.np.algos import MOD_CEM_SSD
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import BatchSampler
from garage.envs import GarageEnv
from garage.experiment import LocalRunner
from garage.np.policies import StableSpringDamperPolicy
from gym.envs.mujoco.block2D import GOAL

def run_task(snapshot_config, *_):
    """Train CEM with Block2D-v1 environment."""
    with LocalRunner(snapshot_config=snapshot_config) as runner:
        env = GarageEnv(env_name='Block2D-v1')
        # K=2 components
        # should be the same as in the mujocopy file block2D.py
        # GOAL = np.array([0.4, -0.1])
        policy = StableSpringDamperPolicy(
                                      env.spec,
                                      GOAL,
                                        K=2,
                                        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        # n_samples = 20
        n_samples = 20 # number of samples in an epoch in CEM

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
                   # best_frac=0.05,
                   best_frac=0.3,
                   max_path_length=100,
                   n_samples=n_samples,
                   init_cov=1.,
                   init_pd_gain=1,
                   elite=True,
                   temperature = .1,
                   entropy_const=20e-2,
                   entropy_step_v=100,
                           )
        # ***important change T in block2D.py (reward def) equal to max_path_length***
        runner.setup(algo, env, sampler_cls=BatchSampler)
        # NOTE: make sure that n_epoch_cycles == n_samples !
        # TODO: it is not clear why the above is required
        # runner.train(n_epochs=100, batch_size=1000, n_epoch_cycles=n_samples, plot=True, store_paths=True)
        runner.train(n_epochs=30, batch_size=100, n_epoch_cycles=n_samples, plot=True, store_paths=False)

# IMPORTANT: change the log directory in batch_polopt.py
run_experiment(
    run_task,
    snapshot_mode='last',
    # seed=1,
    plot=True,
    exp_name='mod_cem_block_1e-1KH',
    exp_prefix='exp',
    log_dir=None,
)

