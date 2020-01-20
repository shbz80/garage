"""Policy based on the paper
Modeling robot discrete movements with state-varying stiffness and damping
in Cartesian space
"""
import akro
import numpy as np
from garage.np.policies.stable_spring_damper import StableSpringDamperPolicy
from YumiKinematics import YumiKinematics

class StableCartSpringDamperPolicy(StableSpringDamperPolicy):
    def __init__(self, env_spec, goal, yumikinparams, K=1):
        assert isinstance(env_spec.action_space, akro.Box)
        assert(K>=0)
        super().__init__(env_spec)
        self.obs_dim = env_spec.observation_space.flat_dim
        assert(self.obs_dim%2 == 0)
        self.dj = self.obs_dim/2
        self.action_dim = env_spec.action_space.flat_dim
        assert(self.obs_dim/2 == self.action_dim)
        self.K = K
        self.ds = 6
        assert(goal.shape==(4,4))
        self.goal = goal
        self.kinparams = yumikinparams
        self.yumiKin = YumiKinematics(yumikinparams)
        self.params = {}
        self.params['base'] = []
        self.params['comp'] = []
        self.initialized = False

    @property
    def vectorized(self):
        """Vectorized or not."""
        return False

    def to_cart_state(self, joint_state):
        x = self.yumiKin.forward(joint_state)
        return x.reshape(-1)

    def to_joint_action(self, cart_action):
        None
        

    def get_action(self, observation):
        """Get action from the policy."""
        assert(self.initialized==True)
        ds = self.ds
        K = self.K
        assert (observation.shape == (ds * 2,))
        s = observation[:ds]-self.goal
        s_dot = observation[ds:]

        base_params = self.params['base']
        component_params = self.params['comp']
        S0 = base_params[0]['S']
        D0 = base_params[0]['D']

        trq_S_comp = np.zeros(ds)
        trq_D_comp = np.zeros(ds)
        for k in range(K):
            Sk = component_params[k]['S']
            Dk = component_params[k]['D']
            muk = component_params[k]['mu']
            lk = component_params[k]['l']
            alphak = s.dot(Sk.dot(s-2.0*muk))
            alphak = np.clip(alphak,0.,None)
            e = -0.25*lk*(alphak**2)
            betak = np.exp(e)
            wk = alphak*betak
            trq_S_comp += wk*Sk.dot(s-muk)
            trq_D_comp += wk*Dk.dot(s_dot)

        trq_S_base = S0.dot(s)
        trq_D_base = D0.dot(s_dot)

        action_trq = -trq_S_base-trq_D_base-trq_S_comp-trq_D_comp
        assert(action_trq.shape==(ds,))
        return action_trq, dict(mean=action_trq, log_std=0)

    def get_actions(self, observations):
        """Get actions from the policy."""
        assert(isinstance(observations, list))
        N = len(observations)
        actions = np.zeros((N, self.ds))
        # actions_info = np.zeros((N, self.ds))
        for n in range(N):
            observation = observations[n]
            actions[n] = self.get_action(observation)
        return actions, dict(mean=actions, log_std=np.zeros((N,self.ds)))

    def get_param_values(self):
        """Get the trainable variables."""
        # return self.component_params, self.base_params
        assert (self.initialized == True)
        return self.params

    def set_param_values(self, params):
        """Get the trainable variables."""
        K = self.K
        ds = self.ds
        b_params = params['base']
        assert(isinstance(b_params, list))
        assert(len(b_params)==1)
        S0 = b_params[0]['S']
        D0 = b_params[0]['D']
        assert (S0.shape == D0.shape == (ds, ds))
        assert (np.all(np.linalg.eigvals(S0) > 0))
        assert (np.all(np.linalg.eigvals(D0) > 0))
        self.params['base'] = b_params.copy()

        c_params = params['comp']
        assert (isinstance(c_params, list))
        assert (len(c_params) == K)
        for k in range(K):
            Sk = c_params[k]['S']
            Dk = c_params[k]['D']
            muk = c_params[k]['mu']
            lk = c_params[k]['l']
            assert(Sk.shape == Dk.shape == (ds,ds))
            assert(np.all(np.linalg.eigvals(Sk) > 0))
            assert (np.all(np.linalg.eigvals(Dk) > 0))
            assert(muk.shape == (ds,))
            assert(lk>0)
        self.params['comp'] = c_params.copy()
        self.initialized = True
        return
