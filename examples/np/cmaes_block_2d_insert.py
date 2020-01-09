#!/usr/bin/env python3
"""This is an example to train a task with CMA-ES.

Here it runs Block2D-v1 environment with 100 epoches.

"""
from garage.experiment import run_experiment
from garage.np.algos import CMAES
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import OnPolicyVectorizedSampler
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianMLPPolicy


def run_task(snapshot_config, *_):
    """Train CMA_ES with Cartpole-v1 environment."""
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        env = TfEnv(env_name='Block2D-v1')

        policy = GaussianMLPPolicy(name='policy',
                                   env_spec=env.spec,
                                   hidden_sizes=(8,),
                                   learn_std=False,
                                   init_std=0.,
                                   )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        n_samples = 15

        algo = CMAES(env_spec=env.spec,
                     policy=policy,
                     baseline=baseline,
                     max_path_length=100,
                     n_samples=n_samples)

        runner.setup(algo, env, sampler_cls=OnPolicyVectorizedSampler)
        # NOTE: make sure that n_epoch_cycles == n_samples !
        runner.train(n_epochs=30, batch_size=100, n_epoch_cycles=n_samples,plot=True,)

# IMPORTANT: change the log directory in batch_polopt.py
run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
    plot=True,
    exp_name='cmaes_block_2d',
    exp_prefix='exp',
    log_dir=None,
)
