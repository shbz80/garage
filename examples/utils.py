from __future__ import print_function
import numpy as np
# import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
# import matplotlib.colors as plt_colors
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.transforms as transforms

def plot_ellipse(ax, mu, sigma, color="k"):
    """
    Based on
   https://matplotlib.org/gallery/statistics/confidence_ellipse.html?highlight=plot%20confidence%20ellipse%20two%20dimensional%20dataset
    """
    cov = sigma
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      ec=color, fill=False)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    n_std = 2
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mu[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mu[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)


def set_params(params, K):
    mu_vec = params[0]
    mu_vec = mu_vec.reshape(K, -1)
    pd_mat = params[1]
    l_vec = params[2]

    params_ = {}
    params_['base'] = []
    params_['base'].append({})
    params_['base'][0]['S'] = pd_mat['base'][0]['S']
    params_['base'][0]['D'] = pd_mat['base'][0]['D']

    params_['comp'] = []
    for k in range(K):
        params_['comp'].append({})
        params_['comp'][k]['S'] = pd_mat['comp'][k]['S']
        params_['comp'][k]['D'] = pd_mat['comp'][k]['D']
        params_['comp'][k]['mu'] = mu_vec[k]
        params_['comp'][k]['l'] = l_vec[k]

    return params_


def iMOGIC_energy_blocks(s, s_dot, params, K, M=2):
    params = set_params(params, K)
    base_params = params['base']
    component_params = params['comp']
    S0 = base_params[0]['S']
    component_stiff_potential = 0
    for k in range(K):
        Sk = component_params[k]['S']
        muk = component_params[k]['mu']
        lk = component_params[k]['l']
        alphak = s.dot(Sk.dot(s - 2.0 * muk))
        alphak = np.clip(alphak, 0., None)
        e = -0.25 * lk * (alphak ** 2)
        betak = np.exp(e)
        component_stiff_potential = component_stiff_potential + (1.0-betak)/lk
    kinectic_energy = 0.5*M*s_dot.dot(s_dot)
    global_stiff_potential = 0.5*s.dot(S0.dot(s))
    potential_energy = global_stiff_potential + component_stiff_potential
    energy = potential_energy + kinectic_energy
    return energy

def iMOGIC_energy_block_vec(S, S_dot, params, K, M=2):
    N = S.shape[0]
    E = np.zeros(N)

    for n in range(N):
        E[n] = iMOGIC_energy_blocks(S[n], S_dot[n], params, K, M=M)

    return E

def iMOGIC_energy_yumi(s, s_dot, params, K, M):
    params = set_params(params, K)
    base_params = params['base']
    component_params = params['comp']
    S0 = base_params[0]['S']
    component_stiff_potential = 0
    for k in range(K):
        Sk = component_params[k]['S']
        muk = component_params[k]['mu']
        lk = component_params[k]['l']
        alphak = s.dot(Sk.dot(s - 2.0 * muk))
        alphak = np.clip(alphak, 0., None)
        e = -0.25 * lk * (alphak ** 2)
        betak = np.exp(e)
        component_stiff_potential = component_stiff_potential + (1.0-betak)/lk
    kinectic_energy = 0.5*s_dot.dot(M.dot(s_dot))
    global_stiff_potential = 0.5*s.dot(S0.dot(s))
    potential_energy = global_stiff_potential + component_stiff_potential
    energy = potential_energy + kinectic_energy
    return energy

def iMOGIC_energy_yumi_vec(S, S_dot, params, K, yumikin):
    N = S.shape[0]
    E = np.zeros(N)

    for n in range(N):
        s = S[n]
        M = yumikin.get_cart_intertia_d(s)
        E[n] = iMOGIC_energy_blocks(s, S_dot[n], params, K, M)

    return E

def iMOGIC_VIC_point(s, s_dot, params, K):
    dS = s.shape[0]
    params = set_params(params, K)
    base_params = params['base']
    component_params = params['comp']
    S0 = base_params[0]['S']
    D0 = base_params[0]['D']
    S = S0
    D = D0
    muS = np.zeros(dS)
    for k in range(K):
        Sk = component_params[k]['S']
        Dk = component_params[k]['D']
        muk = component_params[k]['mu']
        lk = component_params[k]['l']
        alphak = s.dot(Sk.dot(s - 2.0 * muk))
        alphak = np.clip(alphak, 0., None)
        e = -0.25 * lk * (alphak ** 2)
        betak = np.exp(e)
        wk = alphak * betak
        S = S + wk * Sk
        D = D + wk * Dk
        muS = muS + wk * Sk.dot(muk)
    S_inv = np.linalg.pinv(S)
    mu = S_inv.dot(muS)

    return mu, S, D

def iMOGIC_VIC_point_vec(S, S_dot, params, K):
    N = S.shape[0]
    VIC_vec = []

    for n in range(N):
        mu, S_mat, D_mat = iMOGIC_VIC_point(S[n], S_dot[n], params, K)
        VIC_vec.append([mu, S_mat, D_mat])

    return VIC_vec
