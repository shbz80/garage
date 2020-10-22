"""Cross Entropy Method."""
from dowel import logger, tabular
import numpy as np
# from scipy.stats import invwishart
from scipy.stats import wishart
from garage.np.algos import BatchPolopt
from garage.np.policies import StableSpringDamperPolicy
from operator import itemgetter
from garage.misc import tensor_utils
from rewards import process_cart_path_rwd, process_samples_fill
# from garage.np.policies import StableCartSpringDamperPolicy

class MOD_CEM_SSD(BatchPolopt):
    """
    Modified from cem.py
    Modified CEM for stable spring damper (ssd) policy
    """

    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 n_samples,
                 discount=0.99,
                 max_path_length=500,
                 init_cov_diag=np.ones(6),
                 SD_mat_init={},
                 v_scalar_init=3,
                 mu_init=np.zeros(6),
                 best_frac=0.05,
                 extra_std=1.,
                 extra_decay_time=100,
                 elite=True,
                 temperature=0.1,
                 entropy_const = 0.1,
                 entropy_step_v = 100,):
        assert(isinstance(policy, StableSpringDamperPolicy))
        super().__init__(policy, baseline, discount, max_path_length,
                         n_samples)
        self.env_spec = env_spec
        self.n_samples = n_samples
        self.best_frac = best_frac
        self.init_mu_cov = np.diag(np.tile(init_cov_diag, policy.K))
        self.SD_mat_init = SD_mat_init
        self.mu_init = mu_init
        self.init_l_dof = v_scalar_init
        self.extra_std = extra_std
        self.extra_decay_time = extra_decay_time
        self.scope=None
        self.elite = elite
        self.temperature = temperature
        self.KH = entropy_const
        self.epsH_v = entropy_step_v

    def get_beta(self):
        K = self.policy.K

        S0 = self.cur_stat['sd_scale']['base'][0]['S']
        D0 = self.cur_stat['sd_scale']['base'][0]['D']
        S = [self.cur_stat['sd_scale']['comp'][k]['S'] for k in range(K)]
        D = [self.cur_stat['sd_scale']['comp'][k]['D'] for k in range(K)]
        l = self.cur_stat['l_scale']
        v1S0 = self.cur_stat['sd_dof']['base'][0]['S']
        v1D0 = self.cur_stat['sd_dof']['base'][0]['D']
        v1S = np.array([self.cur_stat['sd_dof']['comp'][k]['S'] for k in range(K)])
        v1D = np.array([self.cur_stat['sd_dof']['comp'][k]['D'] for k in range(K)])
        v1l = self.cur_stat['l_dof']
        v2S0 = v1S0 + self.epsH_v
        v2D0 = v1D0 + self.epsH_v
        v2S = v1S + self.epsH_v
        v2D = v1D + self.epsH_v
        v2l = v1l + self.epsH_v

        beta = {}
        deltaH_S0 = wishart.entropy(v2S0, S0/v2S0) - wishart.entropy(v1S0, S0/v1S0)
        beta['betaS0'] = -np.log(v2S0 / v1S0)/deltaH_S0
        deltaH_D0 = wishart.entropy(v2D0, D0/v2D0) - wishart.entropy(v1D0, D0/v1D0)
        beta['betaD0'] = -np.log(v2D0 / v1D0) / deltaH_D0
        betaS = np.zeros(K)
        betaD = np.zeros(K)
        betal = np.zeros(K)
        for k in range(K):
            deltaH_Sk = wishart.entropy(v2S[k], S[k]/v2S[k]) - wishart.entropy(v1S[k], S[k]/v1S[k])
            betaS[k] = -np.log(v2S[k] / v1S[k]) / deltaH_Sk
            deltaH_Dk = wishart.entropy(v2D[k], D[k]/v2D[k]) - wishart.entropy(v1D[k], D[k]/v1D[k])
            betaD[k] = -np.log(v2D[k] / v1D[k]) / deltaH_Dk
            deltaH_lk = wishart.entropy(v2l[k], l[k]/v2l[k]) - wishart.entropy(v1l[k], l[k]/v1l[k])
            betal[k] = -np.log(v2l[k] / v1l[k]) / deltaH_lk
        beta['betaSk'] = betaS
        beta['betaDk'] = betaD
        beta['betalk'] = betal

        return beta

    def get_dof_update(self, re, rb):
        K = self.policy.K
        assert(re<=0 and rb<=0)
        assert ((re - rb) >= -1e-6)
        v1S0 = self.cur_stat['sd_dof']['base'][0]['S']
        v1D0 = self.cur_stat['sd_dof']['base'][0]['D']
        v1S = np.array([self.cur_stat['sd_dof']['comp'][k]['S'] for k in range(K)])
        v1D = np.array([self.cur_stat['sd_dof']['comp'][k]['D'] for k in range(K)])
        v1l = self.cur_stat['l_dof']
        beta = self.get_beta()

        # vT = 100000
        # v0 = self.init_pd_dof
        # E = 30
        # eta = 1e-10
        # beta_m = beta['betaS0']
        # beta_l = beta['betalk']

        KH = self.KH
        KH_l = KH * beta['betaSk'] / beta['betalk']

        # KH = (np.log(vT)-np.log(v0))/((1-eta)*beta_m*E)
        # KH_l = (np.log(vT) - np.log(v0)) / ((1 - eta) * beta_l[0] * E)

        v2S0 = v1S0*np.exp(-beta['betaS0']*KH*-(1-re/rb))
        v2D0 = v1D0*np.exp(-beta['betaD0']*KH*-(1-re/rb))
        v2S = v1S*np.exp(-beta['betaSk']*KH*-(1-re/rb))
        v2D = v1D*np.exp(-beta['betaDk']*KH*-(1-re/rb))
        v2l = v1l*np.exp(-beta['betalk']*KH_l*-(1-re/rb))

        return v2S0, v2D0, v2S, v2D, v2l

    def check_for_sd_dof_equal(self):
        K = self.policy.K
        dof_list = []
        cur_sd_dof = self.cur_stat['sd_dof']
        v = int(cur_sd_dof['base'][0]['S'])
        dof_list.append(v)
        v = int(cur_sd_dof['base'][0]['D'])
        dof_list.append(v)
        for k in range(K):
            v = int(cur_sd_dof['comp'][k]['S'])
            dof_list.append(v)
            v = int(cur_sd_dof['comp'][k]['D'])
            dof_list.append(v)

        if (len(set(dof_list))!=1):
            print('Warning: Wishart DOF not equal', dof_list)
            # assert all v are the same
        return dof_list[0]

    def check_for_l_dof_equal(self):
        dof_list = list(self.cur_stat['l_dof'])
        if (len(set(dof_list)) != 1):
            print('Warning: Wishart DOF not equal', dof_list)
        return dof_list[0]

    def _sample_params(self):
        K = self.policy.K
        dS = self.policy.dS
        cur_mu_mean = self.cur_stat['mu_mean']
        cur_mu_cov = self.cur_stat['mu_cov']
        cur_sd_scale = self.cur_stat['sd_scale']
        cur_sd_dof = self.check_for_sd_dof_equal()
        cur_l_scale = self.cur_stat['l_scale']
        cur_l_dof = self.check_for_l_dof_equal()

        sample_mu = np.random.multivariate_normal(cur_mu_mean, cur_mu_cov)
        sample_pd_mat = {}
        sample_pd_mat['base'] = []
        sample_pd_mat['comp'] = []
        sample_pd_mat['base'].append({})

        cur_S0_W = cur_sd_scale['base'][0]['S']
        cur_S0_trans_W = cur_S0_W[:3,:3]/self.SD_mat_init['S_trans_s']
        W = wishart.rvs(cur_sd_dof, cur_S0_trans_W / cur_sd_dof)
        S_trans = self.SD_mat_init['S_trans_s'] * W
        cur_S0_rot_W = cur_S0_W[3:, 3:] / self.SD_mat_init['S_rot_s']
        W = wishart.rvs(cur_sd_dof, cur_S0_rot_W / cur_sd_dof)
        S_rot = self.SD_mat_init['S_rot_s'] * W
        sample_pd_mat['base'][0]['S'] = np.block([
            [S_trans, np.zeros((3, 3))],
            [np.zeros((3, 3)), S_rot]
        ])

        cur_D0_W = cur_sd_scale['base'][0]['D']
        cur_D0_trans_W = cur_D0_W[:3, :3] / self.SD_mat_init['D_trans_s']
        W = wishart.rvs(cur_sd_dof, cur_D0_trans_W / cur_sd_dof)
        D_trans = self.SD_mat_init['D_trans_s'] * W
        cur_D0_rot_W = cur_D0_W[3:, 3:] / self.SD_mat_init['D_rot_s']
        W = wishart.rvs(cur_sd_dof, cur_D0_rot_W / cur_sd_dof)
        D_rot = self.SD_mat_init['D_rot_s'] * W
        sample_pd_mat['base'][0]['D'] = np.block([
            [D_trans, np.zeros((3, 3))],
            [np.zeros((3, 3)), D_rot]
        ])

        for k in range(K):
            sample_pd_mat['comp'].append({})
            cur_Sk_W = cur_sd_scale['comp'][k]['S']
            cur_Sk_trans_W = cur_Sk_W[:3, :3] / self.SD_mat_init['S_trans_s']
            W = wishart.rvs(cur_sd_dof, cur_Sk_trans_W / cur_sd_dof)
            S_trans = self.SD_mat_init['S_trans_s'] * W
            cur_Sk_rot_W = cur_Sk_W[3:, 3:] / self.SD_mat_init['S_rot_s']
            W = wishart.rvs(cur_sd_dof, cur_Sk_rot_W / cur_sd_dof)
            S_rot = self.SD_mat_init['S_rot_s'] * W
            sample_pd_mat['comp'][k]['S'] = np.block([
             [S_trans, np.zeros((3, 3))],
             [np.zeros((3, 3)), S_rot]
            ])

            cur_Dk_W = cur_sd_scale['comp'][k]['D']
            cur_Dk_trans_W = cur_Dk_W[:3, :3] / self.SD_mat_init['D_trans_s']
            W = wishart.rvs(cur_sd_dof, cur_Dk_trans_W / cur_sd_dof)
            D_trans = self.SD_mat_init['D_trans_s'] * W
            cur_Dk_rot_W = cur_Dk_W[3:, 3:] / self.SD_mat_init['D_rot_s']
            W = wishart.rvs(cur_sd_dof, cur_Dk_rot_W / cur_sd_dof)
            D_rot = self.SD_mat_init['D_rot_s'] * W
            sample_pd_mat['comp'][k]['D'] = np.block([
             [D_trans, np.zeros((3, 3))],
             [np.zeros((3, 3)), D_rot]
            ])

        sample_l = np.zeros(K)
        for k in range(K):
            sample_l[k] = wishart.rvs(cur_l_dof, cur_l_scale[k]/cur_l_dof)

        sample_params = (sample_mu, sample_pd_mat, sample_l)
        return sample_params


    def get_params(self):
        params = self.policy.get_param_values()
        K = self.policy.K

        mu_list = []
        for k in range(K):
            mu = params['comp'][k]['mu']
            mu_list.append(mu)
        mu_array = np.array(mu_list)
        mu_vec = mu_array.reshape(-1)

        mat = {}
        mat['base'] = []
        mat['comp'] = []
        mat['base'].append({})
        mat['base'][0]['S'] = params['base'][0]['S']
        mat['base'][0]['D'] = params['base'][0]['D']
        for k in range(K):
            mat['comp'].append({})
            mat['comp'][k]['S'] = params['comp'][k]['S']
            mat['comp'][k]['D'] = params['comp'][k]['D']

        l_vec = np.zeros(K)
        for k in range(K):
            l_vec[k] = params['comp'][k]['l']

        return mu_vec, mat, l_vec

    def set_params(self, cur_params):
        K = self.policy.K
        mu_vec = cur_params[0]
        mu_vec = mu_vec.reshape(K, -1)
        pd_mat = cur_params[1]
        l_vec = cur_params[2]

        params = {}
        params['base'] = []
        params['base'].append({})
        params['base'][0]['S'] = pd_mat['base'][0]['S']
        params['base'][0]['D'] = pd_mat['base'][0]['D']

        params['comp'] = []
        for k in range(K):
            params['comp'].append({})
            params['comp'][k]['S'] = pd_mat['comp'][k]['S']
            params['comp'][k]['D'] = pd_mat['comp'][k]['D']
            params['comp'][k]['mu'] = mu_vec[k]
            params['comp'][k]['l'] = l_vec[k]

        self.policy.set_param_values(params)

    def update_stat(self):
        K = self.policy.K
        dS = self.policy.dS
        lmda = self.temperature

        all_rtns = np.array(self.all_returns)
        all_params = self.all_params

        M = len(all_rtns)
        weights = np.zeros(M)
        if self.elite:
            best_inds = list(np.argsort(-all_rtns)[:self.n_best])
            rest_inds = list(np.argsort(-all_rtns)[self.n_best:])
            weights[best_inds] = 1./self.n_best
            # weights[best_inds] = all_rtns[best_inds]/np.sum(all_rtns[best_inds])
            weights[rest_inds] = 0.
        else:
            max_all_rtns = np.max(all_rtns)
            all_rtns_diff = all_rtns - max_all_rtns
            all_e = lmda*all_rtns_diff
            weights_unnorm = np.exp(all_e)
            weights = weights_unnorm / np.sum(weights_unnorm)
        assert(np.sum(weights)-1 < 1e-6)

        avg_rtns = np.average(all_rtns)
        avg_best_rtns = np.average(all_rtns, weights=weights)

        assert(isinstance(all_params, list))
        all_mu = np.zeros((M,K*dS))
        all_S0 = np.zeros((M,dS,dS))
        all_D0 = np.zeros((M, dS, dS))
        all_Sk = np.zeros((M, K, dS, dS))
        all_Dk = np.zeros((M, K, dS, dS))
        all_lk = np.zeros((M,K))
        for i in range(M):
            all_mu[i] = all_params[i][0]
            all_S0[i] = all_params[i][1]['base'][0]['S']
            all_D0[i] = all_params[i][1]['base'][0]['D']
            all_Sk[i] = np.array([all_params[i][1]['comp'][j]['S'] for j in range(K)])
            all_Dk[i] = np.array([all_params[i][1]['comp'][j]['D'] for j in range(K)])
            all_lk[i] = all_params[i][2]

        v2S0, v2D0, v2S, v2D, v2l = self.get_dof_update(avg_best_rtns,avg_rtns)

        print('Wishart dof', v2S0, v2D0, v2S, v2D, v2l)

        self.cur_stat['mu_mean'] = np.average(all_mu, axis=0, weights=weights)
        self.cur_stat['mu_cov'] = np.cov(all_mu, rowvar=False, ddof=None, aweights=weights)
        self.cur_stat['l_scale'] = np.average(all_lk, axis=0, weights=weights)
        # self.cur_stat['l_dof'] += int(self.n_best)
        self.cur_stat['l_dof'] = v2l
        self.cur_stat['sd_scale']['base'][0]['S'] = np.average(all_S0, axis=0, weights=weights)
        self.cur_stat['sd_scale']['base'][0]['D'] = np.average(all_D0, axis=0, weights=weights)
        # self.cur_stat['sd_dof']['base'][0]['S'] += int(self.n_best)
        # self.cur_stat['sd_dof']['base'][0]['D'] += int(self.n_best)
        self.cur_stat['sd_dof']['base'][0]['S'] = v2S0
        self.cur_stat['sd_dof']['base'][0]['D'] = v2D0
        mean_all_Sk = np.average(all_Sk, axis=0, weights=weights)
        mean_all_Dk = np.average(all_Dk, axis=0, weights=weights)
        for k in range(K):
            self.cur_stat['sd_scale']['comp'][k]['S'] = mean_all_Sk[k]
            self.cur_stat['sd_scale']['comp'][k]['D'] = mean_all_Dk[k]
            # self.cur_stat['sd_dof']['comp'][k]['S'] += int(self.n_best)
            # self.cur_stat['sd_dof']['comp'][k]['D'] += int(self.n_best)
            self.cur_stat['sd_dof']['comp'][k]['S'] = v2S[k]
            self.cur_stat['sd_dof']['comp'][k]['D'] = v2D[k]

    def train(self, runner):
        """Initialize variables and start training.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        Returns:
            The average return in last epoch cycle.

        """
        # todo
        # import pickle
        # param_file_peg_winit = '/home/shahbaz/Software/garage/examples/np/data/local/peg-winit-full-imped/2/exp_param.pkl'
        # infile = open(param_file_peg_winit, 'rb')
        # peg_winit_param = pickle.load(infile)
        # infile.close()
        # last_ep = peg_winit_param[0]
        # last_param = last_ep['epoc_params'][1]

        dS = self.policy.dS
        K = self.policy.K
        goal = np.zeros(dS)
        # epoch-wise
        init_params = {}
        init_params['base'] = []
        init_params['comp'] = []
        init_params['base'].append({})
        M_init = self.SD_mat_init['M_init'] * np.eye(3)
        S_trans_init = self.SD_mat_init['S_trans_s'] * M_init
        S_rot_init = self.SD_mat_init['S_rot_s'] * M_init
        D_trans_init = self.SD_mat_init['D_trans_s'] * M_init
        D_rot_init = self.SD_mat_init['D_rot_s'] * M_init
        S_init = np.block([[S_trans_init, np.zeros((3,3))],
                          [np.zeros((3,3)), S_rot_init]])
        D_init = np.block([[D_trans_init, np.zeros((3,3))],
                          [np.zeros((3,3)), D_rot_init]])
        init_params['base'][0]['S'] = S_init
        init_params['base'][0]['D'] = D_init
        for k in range(K):
            init_params['comp'].append({})
            init_params['comp'][k]['S'] = S_init/self.SD_mat_init['local_scale']
            init_params['comp'][k]['D'] = D_init/np.sqrt(self.SD_mat_init['local_scale'])
            init_params['comp'][k]['l'] = 1
            init_params['comp'][k]['mu'] = self.mu_init

        self.policy.set_param_values(init_params) #todo
        # self.set_params(last_param)

        self.init_mu_mean, self.init_sd_scale, self.init_l_scale = self.get_params()
        self.cur_stat = {}
        self.cur_stat['mu_mean'] = self.init_mu_mean
        self.cur_stat['sd_scale'] = self.init_sd_scale
        self.cur_stat['l_scale'] = self.init_l_scale
        self.cur_stat['mu_cov'] = self.init_mu_cov
        self.cur_stat['sd_dof'] = {}
        self.cur_stat['sd_dof']['base'] = []
        self.cur_stat['sd_dof']['base'].append({})
        self.cur_stat['sd_dof']['base'][0]['S'] = self.SD_mat_init['v']
        self.cur_stat['sd_dof']['base'][0]['D'] = self.SD_mat_init['v']
        self.cur_stat['sd_dof']['comp'] = []
        for k in range(K):
            self.cur_stat['sd_dof']['comp'].append({})
            self.cur_stat['sd_dof']['comp'][k]['S'] = self.SD_mat_init['v']
            self.cur_stat['sd_dof']['comp'][k]['D'] = self.SD_mat_init['v']

        self.cur_stat['l_dof'] = np.ones(K) * self.init_l_dof

        # epoch-cycle-wise
        self.cur_params = (self.init_mu_mean, self.init_sd_scale, self.init_l_scale)
        self.all_returns = []
        self.all_params = [self.cur_params]
        # constant
        self.n_best = int(self.n_samples * self.best_frac)
        assert self.n_best >= 1, (
            'n_samples is too low. Make sure that n_samples * best_frac >= 1')

        return super().train(runner)

    def train_once(self, itr, path):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        """
        path = self.process_sample(path[0])
        undiscounted_return = sum(path['rewards'])
        rtn = np.mean(undiscounted_return)

        epoch = itr // self.n_samples
        i_sample = itr - epoch * self.n_samples
        tabular.record('Epoch', epoch)
        tabular.record('# Sample', i_sample)
        # -- Stage: Process path
        # rtn = paths['average_return']
        # self.all_returns.append(paths['average_return'])
        self.all_returns.append(rtn)

        # -- Stage: Update policy distribution.
        if (itr + 1) % self.n_samples == 0:
            self.update_stat()
            # print('Params', self.cur_params)
            # Clear for next epoch
            rtn = max(self.all_returns)
            print('mean returns', np.mean(self.all_returns))
            print('std returns', np.std(self.all_returns))
            self.all_returns.clear()
            self.all_params.clear()

        # -- Stage: Generate a new policy for next path sampling
        self.cur_params = self._sample_params() #todo
        self.all_params.append(self.cur_params)
        self.set_params(self.cur_params)

        logger.log(tabular)
        return rtn

    def process_sample(self, path):
        """Return processed sample data based on the collected paths.
        """
        #TODO: comment the following line for MujoCo exp
        # path = process_samples_fill(path, self.policy.T)
        path = process_cart_path_rwd(path, self.policy.yumiKin, self.discount)
        return path