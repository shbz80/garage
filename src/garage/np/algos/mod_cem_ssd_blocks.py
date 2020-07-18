"""Cross Entropy Method."""
from dowel import logger, tabular
import numpy as np
from scipy.stats import wishart
# from garage.np.algos import BatchPolopt
from garage.np.policies import StableSpringDamperPolicy
# from operator import itemgetter
from garage.misc import tensor_utils
# from rewards import process_cart_path_rwd, process_samples_fill
# from garage.np.policies import StableCartSpringDamperPolicy
from garage.np.algos import MOD_CEM_SSD

class MOD_CEM_SSD_BLOCKS(MOD_CEM_SSD):
    """
    Modified from cem.py
    Modified CEM for stable spring damper (ssd) policy
    """
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
        cur_S0_W = cur_S0_W/self.SD_mat_init['S_s']
        W = wishart.rvs(cur_sd_dof, cur_S0_W / cur_sd_dof)
        S = self.SD_mat_init['S_s'] * W
        sample_pd_mat['base'][0]['S'] = S

        cur_D0_W = cur_sd_scale['base'][0]['D']
        cur_D0_W = cur_D0_W / self.SD_mat_init['D_s']
        W = wishart.rvs(cur_sd_dof, cur_D0_W / cur_sd_dof)
        D = self.SD_mat_init['D_s'] * W
        sample_pd_mat['base'][0]['D'] = D

        for k in range(K):
            sample_pd_mat['comp'].append({})
            cur_Sk_W = cur_sd_scale['comp'][k]['S']
            cur_Sk_W = cur_Sk_W / self.SD_mat_init['S_s']
            W = wishart.rvs(cur_sd_dof, cur_Sk_W / cur_sd_dof)
            S = self.SD_mat_init['S_s'] * W
            sample_pd_mat['comp'][k]['S'] = S

            cur_Dk_W = cur_sd_scale['comp'][k]['D']
            cur_Dk_W = cur_Dk_W / self.SD_mat_init['D_s']
            W = wishart.rvs(cur_sd_dof, cur_Dk_W / cur_sd_dof)
            D = self.SD_mat_init['D_s'] * W
            sample_pd_mat['comp'][k]['D'] = D

        sample_l = np.zeros(K)
        for k in range(K):
            sample_l[k] = wishart.rvs(cur_l_dof, cur_l_scale[k]/cur_l_dof)

        sample_params = (sample_mu, sample_pd_mat, sample_l)
        return sample_params


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
        # param_file_peg_winit = '/home/shahbaz/Software/garage/examples/np/data/local/blocks-initpos2-K8/2/exp_param.pkl'
        # infile = open(param_file_peg_winit, 'rb')
        # peg_winit_param = pickle.load(infile)
        # infile.close()
        # epoch = 4
        # last_ep = peg_winit_param[epoch]  # select epoch
        # sample_ = 0
        # last_param = last_ep['epoc_params'][sample_]  # select sample from epoch

        # last_param = last_ep['epoc_params'][0]  # select sample from epoch
        # last_ep = peg_winit_param[15] # success epoch
        # last_param = last_ep['epoc_params'][2]  # select sample from success epoch

        dS = self.policy.dS
        K = self.policy.K
        goal = np.zeros(dS)
        # epoch-wise
        init_params = {}
        init_params['base'] = []
        init_params['comp'] = []
        init_params['base'].append({})
        M_init = self.SD_mat_init['M_init'] * np.eye(dS)
        S_init = self.SD_mat_init['S_s'] * M_init
        D_init = self.SD_mat_init['D_s'] * M_init

        init_params['base'][0]['S'] = S_init
        init_params['base'][0]['D'] = D_init
        for k in range(K):
            init_params['comp'].append({})
            init_params['comp'][k]['S'] = S_init/self.SD_mat_init['local_scale']
            init_params['comp'][k]['D'] = D_init/np.sqrt(self.SD_mat_init['local_scale'])
            init_params['comp'][k]['l'] = 1
            init_params['comp'][k]['mu'] = self.mu_init

        self.policy.set_param_values(init_params)
        # self.set_params(last_param) #todo

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

        return super(MOD_CEM_SSD, self).train(runner)

    def process_sample(self, path):
        """Return processed sample data based on the collected paths.
        """
        path['returns'] = tensor_utils.discount_cumsum(path['rewards'], self.discount)
        return path
