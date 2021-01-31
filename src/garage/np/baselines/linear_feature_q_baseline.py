"""A linear value function (baseline) based on features."""
import numpy as np

from garage.np.baselines import LinearFeatureBaseline


class LinearFeatureQBaseline(LinearFeatureBaseline):
    """A linear Q value function (baseline) based on features.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        reg_coeff (float): Regularization coefficient.
        name (str): Name of baseline.

    """

    def __init__(self, env_spec, reg_coeff=1e-5, name='LinearFeatureQBaseline'):
        super().__init__(env_spec, reg_coeff, name)


    def _features(self, path):
        """Extract features from path.

        Args:
            path (list[dict]): Sample paths.

        Returns:
            numpy.ndarray: Extracted features.

        """
        obs = np.clip(path['observations'], self.lower_bound, self.upper_bound)
        acts = np.clip(path['actions'], self.lower_bound, self.upper_bound)
        obs_acts = np.concatenate([obs,acts],axis=1)
        length = len(path['rewards'])
        al = np.arange(length).reshape(-1, 1) / 100.0
        return np.concatenate(
            [obs_acts, obs_acts**2, al, al**2, al**3,
             np.ones((length, 1))], axis=1)

