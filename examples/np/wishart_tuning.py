import numpy as np
import matplotlib.pyplot as plt
# from utils import plot_ellipse
from scipy.stats import wishart


D = 3
num_samples = 10000
min = 20/200.
mean = 80/200.
max = 200/200.


for v in range(D+2, 10000, 5):
    print('v', v)
    num_greater = 0
    num_lesser = 0
    Sig = np.eye(D) * mean / float(v)
    for n in range(num_samples):
        W = wishart.rvs(v, Sig)
        W_max = np.full((D,), max)
        W_min = np.full((D,), min)
        if np.any(np.diag(W)>W_max):
            num_greater += 1
        if np.any(np.diag(W)<W_min):
            num_lesser += 1
    print('W', W)
    if num_greater>0 or num_lesser>0:
        print('num_greater', num_greater)
        print('num_lesser', num_lesser)


