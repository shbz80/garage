#!/usr/bin/env python3
"""An example to train a task with TRPO algorithm."""
import os

import gym
from garage.experiment import run_experiment
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianMLPPolicy


def run_task(snapshot_config, *_):
    """Run task."""
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        env = TfEnv(gym.make('Block2D-v1'))

        policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(8,))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    max_path_length=100,
                    discount=0.99,
                    max_kl_step=0.05)

        runner.setup(algo, env)
        runner.train(n_epochs=100, batch_size=1500, plot=True)


run_experiment(
    run_task,
    snapshot_mode='last',
    # seed=1,
    plot=True,
    exp_name='trpo_block_2d',
    exp_prefix='exp',
    log_dir=None,
)
