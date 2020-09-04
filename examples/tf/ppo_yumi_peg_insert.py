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


def run_task(snapshot_config, *_):
    """Run task."""
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        env = TfEnv(gym.make('YumiPeg-v1'))

        policy = GaussianMLPPolicy(env_spec=env.spec,
                                   hidden_sizes=(32,32),
                                   learn_std=True,
                                   adaptive_std=False,
                                   std_share_network=False,
                                   init_std=1.,
                                   )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = PPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    max_path_length=100,
                   gae_lambda=0.95,
                   lr_clip_range=0.2,
                    discount=0.99,
                    )

        runner.setup(algo, env)
        runner.train(n_epochs=100, batch_size=5000, plot=True)


run_experiment(
    run_task,
    snapshot_mode='last',
    # seed=1,
    plot=True,
    exp_name='ppo_yumi_peg_std1_bs5000',
    exp_prefix='exp',
    log_dir=None,
)
