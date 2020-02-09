"""Cross Entropy Method."""
from dowel import logger, tabular
import numpy as np
# from scipy.stats import invwishart
from scipy.stats import wishart
from garage.np.algos import BatchPolopt
from garage.np.policies import StableSpringDamperPolicy
from operator import itemgetter
from garage.misc import tensor_utils
from rewards import process_cart_path_rwd
from garage.np.policies import StableCartSpringDamperPolicy

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
                 init_cov=1.,
                 init_pd_gain=1.,
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
        self.init_mu_cov = init_cov * np.eye(policy.K * policy.dS)
        self.init_pd_gain = init_pd_gain
        self.init_pd_dof = policy.dS + 2.
        self.init_l_dof = 3.         # D=1
        # self.init_pd_dof = policy.dS
        # self.init_l_dof = 1         # D=1
        self.best_frac = best_frac
        self.extra_std = extra_std
        self.extra_decay_time = extra_decay_time
        self.scope=None
        self.elite = elite
        self.temperature = temperature
        self.KH = entropy_const
        self.epsH_v = entropy_step_v

    def get_beta(self):
        K = self.policy.K

        S0 = self.cur_stat['pd_scale']['base'][0]['S']
        D0 = self.cur_stat['pd_scale']['base'][0]['D']
        S = [self.cur_stat['pd_scale']['comp'][k]['S'] for k in range(K)]
        D = [self.cur_stat['pd_scale']['comp'][k]['D'] for k in range(K)]
        l = self.cur_stat['l_scale']
        v1S0 = self.cur_stat['pd_dof']['base'][0]['S']
        v1D0 = self.cur_stat['pd_dof']['base'][0]['D']
        v1S = np.array([self.cur_stat['pd_dof']['comp'][k]['S'] for k in range(K)])
        v1D = np.array([self.cur_stat['pd_dof']['comp'][k]['D'] for k in range(K)])
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
        assert(re>=rb)
        v1S0 = self.cur_stat['pd_dof']['base'][0]['S']
        v1D0 = self.cur_stat['pd_dof']['base'][0]['D']
        v1S = np.array([self.cur_stat['pd_dof']['comp'][k]['S'] for k in range(K)])
        v1D = np.array([self.cur_stat['pd_dof']['comp'][k]['D'] for k in range(K)])
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


    def _sample_params(self):
        K = self.policy.K
        dS = self.policy.dS
        cur_mu_mean = self.cur_stat['mu_mean']
        cur_mu_cov = self.cur_stat['mu_cov']
        cur_pd_scale = self.cur_stat['pd_scale']
        cur_pd_dof = self.cur_stat['pd_dof']
        cur_l_scale = self.cur_stat['l_scale']
        cur_l_dof = self.cur_stat['l_dof']

        sample_mu = np.random.multivariate_normal(cur_mu_mean, cur_mu_cov)
        sample_pd_mat = {}
        sample_pd_mat['base'] = []
        sample_pd_mat['comp'] = []
        sample_pd_mat['base'].append({})
        cur_S0_cov = cur_pd_scale['base'][0]['S']
        cur_S0_dof = int(cur_pd_dof['base'][0]['S'])
        sample_pd_mat['base'][0]['S'] = wishart.rvs(cur_S0_dof, cur_S0_cov/cur_S0_dof)
        # TODO: revert this
        # sample_pd_mat['base'][0]['S'] = cur_S0_cov
        cur_D0_cov = cur_pd_scale['base'][0]['D']
        cur_D0_dof = int(cur_pd_dof['base'][0]['D'])
        sample_pd_mat['base'][0]['D'] = wishart.rvs(cur_D0_dof, cur_D0_cov/cur_D0_dof)
        for k in range(K):
            sample_pd_mat['comp'].append({})
            cur_Sk_cov = cur_pd_scale['comp'][k]['S']
            cur_Sk_dof = int(cur_pd_dof['comp'][k]['S'])
            sample_pd_mat['comp'][k]['S'] = wishart.rvs(cur_Sk_dof, cur_Sk_cov/cur_Sk_dof)
            cur_Dk_cov = cur_pd_scale['comp'][k]['D']
            cur_Dk_dof = int(cur_pd_dof['comp'][k]['D'])
            sample_pd_mat['comp'][k]['D'] = wishart.rvs(cur_Dk_dof, cur_Dk_cov/cur_Dk_dof)

        sample_l = np.zeros(K)
        for k in range(K):
            cur_l_dof_k = int(cur_l_dof[k])
            sample_l[k] = wishart.rvs(cur_l_dof_k, cur_l_scale[k]/cur_l_dof_k)

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

        print(v2S0, v2D0, v2S, v2D, v2l)

        self.cur_stat['mu_mean'] = np.average(all_mu, axis=0, weights=weights)
        self.cur_stat['mu_cov'] = np.cov(all_mu, rowvar=False, ddof=None, aweights=weights)
        self.cur_stat['l_scale'] = np.average(all_lk, axis=0, weights=weights)
        # self.cur_stat['l_dof'] += int(self.n_best)
        self.cur_stat['l_dof'] = v2l
        self.cur_stat['pd_scale']['base'][0]['S'] = np.average(all_S0, axis=0, weights=weights)
        self.cur_stat['pd_scale']['base'][0]['D'] = np.average(all_D0, axis=0, weights=weights)
        # self.cur_stat['pd_dof']['base'][0]['S'] += int(self.n_best)
        # self.cur_stat['pd_dof']['base'][0]['D'] += int(self.n_best)
        self.cur_stat['pd_dof']['base'][0]['S'] = v2S0
        self.cur_stat['pd_dof']['base'][0]['D'] = v2D0
        mean_all_Sk = np.average(all_Sk, axis=0, weights=weights)
        mean_all_Dk = np.average(all_Dk, axis=0, weights=weights)
        for k in range(K):
            self.cur_stat['pd_scale']['comp'][k]['S'] = mean_all_Sk[k]
            self.cur_stat['pd_scale']['comp'][k]['D'] = mean_all_Dk[k]
            # self.cur_stat['pd_dof']['comp'][k]['S'] += int(self.n_best)
            # self.cur_stat['pd_dof']['comp'][k]['D'] += int(self.n_best)
            self.cur_stat['pd_dof']['comp'][k]['S'] = v2S[k]
            self.cur_stat['pd_dof']['comp'][k]['D'] = v2D[k]

    def train(self, runner):
        """Initialize variables and start training.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        Returns:
            The average return in last epoch cycle.

        """
        dS = self.policy.dS
        K = self.policy.K
        a = self.init_pd_gain
        goal = np.zeros(dS)
        # epoch-wise
        init_params = {}
        init_params['base'] = []
        init_params['comp'] = []
        init_params['base'].append({})
        init_params['base'][0]['S'] = a*np.eye(dS)
        init_params['base'][0]['D'] = a*0.5*np.eye(dS)
        for k in range(K):
            init_params['comp'].append({})
            init_params['comp'][k]['S'] = a*np.eye(dS)
            init_params['comp'][k]['D'] = a*np.eye(dS)
            init_params['comp'][k]['l'] = 1
            # init_params['comp'][k]['mu'] = self.policy.goal_cart
            init_params['comp'][k]['mu'] = goal

        self.policy.set_param_values(init_params)

        self.init_mu_mean, self.init_pd_scale, self.init_l_scale = self.get_params()
        self.cur_stat = {}
        self.cur_stat['mu_mean'] = self.init_mu_mean
        self.cur_stat['pd_scale'] = self.init_pd_scale
        self.cur_stat['l_scale'] = self.init_l_scale
        self.cur_stat['mu_cov'] = self.init_mu_cov
        self.cur_stat['pd_dof'] = {}
        self.cur_stat['pd_dof']['base'] = []
        self.cur_stat['pd_dof']['base'].append({})
        self.cur_stat['pd_dof']['base'][0]['S'] = self.init_pd_dof
        self.cur_stat['pd_dof']['base'][0]['D'] = self.init_pd_dof
        self.cur_stat['pd_dof']['comp'] = []
        for k in range(K):
            self.cur_stat['pd_dof']['comp'].append({})
            self.cur_stat['pd_dof']['comp'][k]['S'] = self.init_pd_dof
            self.cur_stat['pd_dof']['comp'][k]['D'] = self.init_pd_dof

        self.cur_stat['l_dof'] = np.ones(K) * self.init_l_dof

        # epoch-cycle-wise
        self.cur_params = (self.init_mu_mean, self.init_pd_scale, self.init_l_scale)
        self.all_returns = []
        self.all_params = [self.cur_params]
        # constant
        self.n_best = int(self.n_samples * self.best_frac)
        assert self.n_best >= 1, (
            'n_samples is too low. Make sure that n_samples * best_frac >= 1')

        return super().train(runner)

    def train_once(self, itr, paths):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        """
        paths = self.process_samples(itr, paths)
        undiscounted_returns = [sum(path['rewards']) for path in paths]
        rtn = np.mean(undiscounted_returns)

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
            print('Params', self.cur_params)
            # Clear for next epoch
            rtn = max(self.all_returns)
            self.all_returns.clear()
            self.all_params.clear()

        # -- Stage: Generate a new policy for next path sampling
        self.cur_params = self._sample_params()
        self.all_params.append(self.cur_params)
        self.set_params(self.cur_params)

        logger.log(tabular)
        return rtn

    def process_samples(self, itr, paths):
        """Return processed sample data based on the collected paths.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths

        Returns:
            dict: Processed sample data, with key
                * average_return: (float)

        """
        # print(len(paths))
        baselines = []
        returns = []

        max_path_length = self.max_path_length

        if hasattr(self.baseline, 'predict_n'):
            all_path_baselines = self.baseline.predict_n(paths)
        else:
            all_path_baselines = [
                self.baseline.predict(path) for path in paths
            ]

        for idx, path in enumerate(paths):
            # baselines
            path['baselines'] = all_path_baselines[idx]
            baselines.append(path['baselines'])

            # returns
            path['returns'] = tensor_utils.discount_cumsum(
                path['rewards'], self.discount)
            returns.append(path['returns'])

        agent_infos = [path['agent_infos'] for path in paths]
        agent_infos = tensor_utils.stack_tensor_dict_list([
            tensor_utils.pad_tensor_dict(p, max_path_length)
            for p in agent_infos
        ])

        valids = [np.ones_like(path['returns']) for path in paths]
        valids = tensor_utils.pad_tensor_n(valids, max_path_length)

        average_discounted_return = (np.mean(
            [path['returns'][0] for path in paths]))

        undiscounted_returns = [sum(path['rewards']) for path in paths]
        self.episode_reward_mean.extend(undiscounted_returns)

        # TODO
        # ent = np.sum(self.policy.distribution.entropy(agent_infos) *
        #              valids) / np.sum(valids)

        # samples_data = dict(average_return=np.mean(undiscounted_returns))

        tabular.record('Iteration', itr)
        tabular.record('AverageDiscountedReturn', average_discounted_return)
        tabular.record('AverageReturn', np.mean(undiscounted_returns))
        tabular.record('Extras/EpisodeRewardMean',
                       np.mean(self.episode_reward_mean))
        tabular.record('NumTrajs', len(paths))
        # tabular.record('Entropy', ent)
        # tabular.record('Perplexity', np.exp(ent))
        tabular.record('StdReturn', np.std(undiscounted_returns))
        tabular.record('MaxReturn', np.max(undiscounted_returns))
        tabular.record('MinReturn', np.min(undiscounted_returns))
        if isinstance(self.policy, StableCartSpringDamperPolicy):
            # only one path per parameter sample
            process_cart_path_rwd(paths[0], self.policy.yumiKin, self.discount)
        return paths
