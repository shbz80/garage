"""Policies which use NumPy as a numerical backend."""

from garage.np.policies.fixed_policy import FixedPolicy
from garage.np.policies.policy import Policy
from garage.np.policies.scripted_policy import ScriptedPolicy
# from garage.np.policies.stable_cart_spring_damper import StableSpringDamperPolicy
# from garage.np.policies.stable_cart_spring_damper import StableCartSpringDamperPolicy
__all__ = [
    'FixedPolicy',
    'Policy',
    'ScriptedPolicy',
    # 'StableSpringDamperPolicy',
    # 'StableCartSpringDamperPolicy',
]
