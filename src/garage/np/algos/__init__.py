"""Reinforcement learning algorithms which use NumPy as a numerical backend."""
from garage.np.algos.cem import CEM
from garage.np.algos.cma_es import CMAES
from garage.np.algos.meta_rl_algorithm import MetaRLAlgorithm
from garage.np.algos.nop import NOP
from garage.np.algos.rl_algorithm import RLAlgorithm
#from garage.np.algos.mod_cem_ssd import MOD_CEM_SSD

__all__ = [
    'RLAlgorithm',
    'CEM',
    'CMAES',
    'MetaRLAlgorithm',
    'NOP',
]
