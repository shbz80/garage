#!/usr/bin/env python3
"""An example to train a task with TRPO algorithm."""
import os

import gym
from garage.experiment import run_experiment
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import PPO
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianMLPPolicy
import tensorflow as tf
T = 200
N = 30
def run_task(snapshot_config, *_):
    """Run task."""
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        env = TfEnv(gym.make('Block2D-v1'))

        policy = GaussianMLPPolicy(env_spec=env.spec,
                                   hidden_sizes=(32,32),
                                   hidden_nonlinearity=tf.nn.tanh,
                                   output_nonlinearity=None,
                                   init_std=2,
                                   )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = PPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    max_path_length=T,
                   gae_lambda=0.95,
                   lr_clip_range=0.2,
                    discount=0.99,
                    # max_kl_step=0.05
                   )

        runner.setup(algo, env)
        runner.train(n_epochs=100, batch_size=N*T, plot=True)


run_experiment(
    run_task,
    snapshot_mode='last',
    # seed=1,
    plot=True,
    exp_name='ppo_block_2d',
    exp_prefix='v35',
    log_dir=None,
)
