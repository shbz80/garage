#!/usr/bin/env python3
"""
This is an example to train a task with Cross Entropy Method.

Here it runs Block2D-v1 environment with 100 epoches.

Results:
    AverageReturn: 100
    RiseTime: epoch 8
"""
from garage.experiment import run_experiment
from garage.np.algos import CEM
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import OnPolicyVectorizedSampler
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianMLPPolicy


def run_task(snapshot_config, *_):
    """Train CEM with Block2D-v1 environment."""
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        env = TfEnv(env_name='Block2D-v1')

        policy = GaussianMLPPolicy(name='policy',
                                      env_spec=env.spec,
                                      hidden_sizes=(8,),
                                    learn_std=False,
                                   init_std=0.,
                                   )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        # n_samples = 20
        n_samples = 15 # number of samples in an epoch in CEM

        # itr is the number of RL iterations consisting of a number of rollouts
        # for CEM when we use deterministic policy then we need only one RL rollout,
        # so in this case one itr consists of only one rollout.
        # number of rollouts is automatically determined from path_length and batch_size
        # we use path_length=100 and batch_size=100
        # In other RL algos one epoch consists of iteration, but in CEM one epoc corresponds
        # to one iteration of CEM that consists of n_samples rollouts.

        algo = CEM(env_spec=env.spec,
                   policy=policy,
                   baseline=baseline,
                   # best_frac=0.05,
                   best_frac=0.3,
                   max_path_length=100,
                   n_samples=n_samples)
        # ***important change T in block2D.py (reward def) equal to max_path_length***
        runner.setup(algo, env, sampler_cls=OnPolicyVectorizedSampler)
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
    exp_name='cem_block_2d',
    exp_prefix='exp',
    log_dir=None,
)

