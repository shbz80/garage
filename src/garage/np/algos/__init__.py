"""Reinforcement learning algorithms which use NumPy as a numerical backend."""
from garage.np.algos.base import RLAlgorithm
from garage.np.algos.batch_polopt import BatchPolopt
from garage.np.algos.cem import CEM
from garage.np.algos.cma_es import CMAES
from garage.np.algos.nop import NOP
from garage.np.algos.off_policy_rl_algorithm import OffPolicyRLAlgorithm
# from garage.np.algos.mod_cem_ssd import MOD_CEM_SSD
# __all__ = [
#     'RLAlgorithm', 'BatchPolopt', 'CEM', 'CMAES', 'NOP', 'OffPolicyRLAlgorithm', 'MOD_CEM_SSD'
# ]
__all__ = [
    'RLAlgorithm', 'BatchPolopt', 'CEM', 'CMAES', 'NOP', 'OffPolicyRLAlgorithm'
]
