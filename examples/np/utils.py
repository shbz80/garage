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

# def gamma_function()
