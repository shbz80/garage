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
        # assert isinstance(env_spec.action_space, akro.Box)
        # assert(K>=0)
        super().__init__(env_spec, goal, K)
        self.obs_dim = env_spec.observation_space.flat_dim
        # assert(self.obs_dim%2 == 0)
        self.dJ = self.obs_dim//2
        # self.action_dim = env_spec.action_space.flat_dim
        # assert(self.obs_dim/2 == self.action_dim)
        # self.K = K
        self.dS = 6
        assert(goal.shape==(7,))
        # self.goal = goal
        self.kinparams = yumikinparams
        self.yumiKin = YumiKinematics(yumikinparams)
        self.goal_cart = self.yumiKin.goal_cart
        # self.params = {}
        # self.params['base'] = []
        # self.params['comp'] = []
        # self.initialized = False
        self.J_Ad_curr = None

    # @property
    # def vectorized(self):
    #     """Vectorized or not."""
    #     return False
    def reset(self, dones=None):
        self.yumiKin = YumiKinematics(self.kinparams)

    def get_cart_state(self, joint_state):
        dJ = self.dJ
        q = joint_state[:dJ]
        q_dot = joint_state[dJ:]
        return self.yumiKin.get_cart_error_frame_terms(q,q_dot)

    def get_action(self, observation):
        """Get action from the policy."""
        assert(self.initialized==True)
        dS = self.dS
        dJ = self.dJ
        K = self.K
        assert (observation.shape == (dJ * 2,))
        x_d_e, x_dot_d_e, J_Ad = self.get_cart_state(observation)
        s = x_d_e
        s_dot = x_dot_d_e
        self.J_Ad_curr = J_Ad

        # s = observation[:dS] - self.goal
        # s_dot = observation[dS:]

        base_params = self.params['base']
        component_params = self.params['comp']
        S0 = base_params[0]['S']
        D0 = base_params[0]['D']

        force_S_comp = np.zeros(dS)
        force_D_comp = np.zeros(dS)
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
            # wk = 0 #TODO: revert this
            force_S_comp += wk*Sk.dot(s-muk)
            force_D_comp += wk*Dk.dot(s_dot)

        # v = 0.2
        force_S_base = S0.dot(s)
        force_D_base = D0.dot(s_dot)

        action_force = -force_S_base-force_D_base-force_S_comp-force_D_comp
        # action_trq = -force_S_base - force_D_base - force_S_comp - force_D_comp
        action_trq = self.J_Ad_curr.T.dot(action_force)
        # print('Trqs:',action_trq)
        assert(action_trq.shape==(dJ,))
        # return action_trq, dict(mean=action_trq, log_std=0)
        return action_trq, dict(mean=action_force, log_std=0)

    def get_actions(self, observations):
        """Get actions from the policy."""
        assert(isinstance(observations, list))
        N = len(observations)
        action_trqs = np.zeros((N, self.dJ))
        action_forces = np.zeros((N, self.dS))
        for n in range(N):
            observation = observations[n]
            action_trq, action_stat = self.get_action(observation)
            action_trqs[n] = action_trq
            action_force = action_stat['mean']
            action_forces[n] = action_force
        return action_trqs, dict(mean=action_forces, log_std=np.zeros((N,self.dS)))

    def get_param_values(self):
        """Get the trainable variables."""
        # return self.component_params, self.base_params
        assert (self.initialized == True)
        return self.params

    def set_param_values(self, params):
        """Get the trainable variables."""
        K = self.K
        dS = self.dS
        b_params = params['base']
        assert(isinstance(b_params, list))
        assert(len(b_params)==1)
        S0 = b_params[0]['S']
        D0 = b_params[0]['D']
        assert (S0.shape == D0.shape == (dS, dS))
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
            assert(Sk.shape == Dk.shape == (dS,dS))
            assert(np.all(np.linalg.eigvals(Sk) > 0))
            assert (np.all(np.linalg.eigvals(Dk) > 0))
            assert(muk.shape == (dS,))
            assert(lk>0)
        self.params['comp'] = c_params.copy()
        self.initialized = True
        return
