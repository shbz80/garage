import pickle
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from gym.envs.mujoco.block2D import GOAL
from utils import iMOGIC_energy_block_vec, iMOGIC_energy_blocks
from utils import iMOGIC_VIC_point_vec
from matplotlib import rc
from utils import plot_ellipse

font_size_1 = 12
font_size_2 = 14
plt.rcParams.update({'font.size': font_size_1})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

############################################
'''
pos2 k8 selected.
k0 2, k1 3, k2 3, k4 4, k8 2, k16 4

'''
############################################

base_np_filename = '/home/shahbaz/Software/garage/examples/np/data/local'
prefix = 'blocks-initpos2-K'

K = 8
exp_name = '2'
plt.rcParams["figure.figsize"] = (6,3)
# fig = plt.figure()
# plt.axis('off')
# plt.rcParams['figure.constrained_layout.use'] = True

filename = base_np_filename + '/' + prefix + str(K) + '/' + exp_name + '/' + 'exp_log.pkl'
infile = open(filename, 'rb')
exp_log = pickle.load(infile)
infile.close()

filename = base_np_filename + '/' + prefix + str(K) + '/' + exp_name + '/' + 'exp_param.pkl'
infile = open(filename, 'rb')
exp_param = pickle.load(infile)
infile.close()

epoch = exp_log[0]
obs0 = epoch[0]['observations']
pos0 = obs0[:,:2] - GOAL
T = pos0.shape[0]
dt = 0.01
tm = np.array(range(T))*dt
vel0 = np.zeros((T,2))
param_ep = exp_param[0]
param0 = param_ep['epoc_params'][0]
assert(K == len(param0[2]))

U = 30
V = 30
s1 = np.linspace(-0.3, 0.2, U)
s2 = np.linspace(-0.7, 0.2, V)
S1, S2 = np.meshgrid(s1, s2)
F = np.zeros((U,V))
for i in range(U):
    for j in range(V):
        S = np.array([S1[i,j], S2[i,j]])
        S_dot = np.zeros(2)
        F[i,j] = iMOGIC_energy_blocks(S, S_dot, param0, K, M=2)

v0 = iMOGIC_energy_block_vec(pos0, vel0, param0, K, M=2)
# v0 = np.zeros(v0.shape)

fig = plt.figure()
# plt.axis('off')
# plt.legend(loc='upper right', bbox_to_anchor=(2., 1.),frameon=False,ncol=3)
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.contour3D(S1, S2, F, 30, label='Energy function', colors='g', alpha=0.7)
# ax.plot_wireframe(S1, S2, F, alpha=.7,label='Energy function')
# ax.plot_surface(S1, S2, F,
#                 cmap='viridis', edgecolor='none',alpha=0.7)
ax.plot3D(pos0[:,0], pos0[:,1], v0, color='b',linewidth=3,label='Trajectory')
ax.scatter(0,0,0,color='r',marker='o',s=20, label='Goal')
ax.set_xlabel(r'$s_1$',fontsize=font_size_2)
ax.set_ylabel(r'$s_2$',fontsize=font_size_2)
ax.set_zlabel(r'$V$',fontsize=font_size_2)
ax.set_title(r'\textbf{(d)}', position=(0.1,0.9), fontsize=font_size_2)

# plt.axis('off')
# ax.grid(b=None)

final_sample = 0
epoch = exp_log[-1]
obsl = epoch[final_sample]['observations']
posl = obsl[:,:2] - GOAL
vell = obsl[:,2:4]
param_ep = exp_param[-1]
paraml = param_ep['epoc_params'][final_sample]
F = np.zeros((U,V))
for i in range(U):
    for j in range(V):
        S = np.array([S1[i,j], S2[i,j]])
        S_dot = np.zeros(2)
        F[i,j] = iMOGIC_energy_blocks(S, S_dot, paraml, K, M=2)
vl = iMOGIC_energy_block_vec(posl, vel0, paraml, K, M=2)
ax = fig.add_subplot(1, 2, 2, projection='3d')
# ax.plot_wireframe(S1, S2, F, alpha=.7,label='Energy function')
ax.contour3D(S1, S2, F, 30, label='Energy function', colors='g', alpha=0.7)
# ax.plot_surface(S1, S2, F,
#                 cmap=cm.coolwarm, edgecolor='none',alpha=1,linewidth=2)
ax.plot3D(posl[:,0], posl[:,1], vl, color='b', linewidth=3, label='Trajectory')
ax.scatter(0,0,0,color='r',marker='o',s=20, label='Goal')
ax.set_xlabel(r'$s_1$',fontsize=font_size_2)
ax.set_ylabel(r'$s_2$',fontsize=font_size_2)
ax.set_zlabel(r'$V$',fontsize=font_size_2)
# ax.legend()
plt.subplots_adjust(left=0.0, bottom=0.1, right=0.92, top=0.98, wspace=0.15, hspace=0)
plt.legend(loc='upper left', bbox_to_anchor=(-.5, 1.02),frameon=False,ncol=3)
ax.set_title(r'\textbf{(e)}', position=(0.1,0.9), fontsize=font_size_2)
# plt.axis('off')
# ax.grid(b=None)

plt.show(block=True)

VIC_vec = iMOGIC_VIC_point_vec(posl, vell, paraml, K)
mu_traj, S_traj, D_traj = zip(*VIC_vec)
posl_ref = np.array(mu_traj)
S_traj = list(S_traj)
D_traj = list(D_traj)


fig = plt.figure(figsize=(6, 2))
ax = fig.add_subplot(1, 3, 1)
ax.set_xlabel(r'$t$',fontsize=font_size_2)
ax.set_ylabel(r'$V$',fontsize=font_size_2,rotation=0)
ax.set_title(r'\textbf{(a)}', position=(-0.3,0.85), fontsize=font_size_2)
ax.plot(tm,v0,color='g',label=r'Itr 0')
ax.plot(tm,vl,color='b',label=r'Itr 99')
ax.legend(frameon=False)

ax = fig.add_subplot(1, 3, 3)
ax.set_xlabel(r'$s_1$',fontsize=font_size_2)
ax.set_ylabel(r'$s_2$',fontsize=font_size_2,rotation=0)
ax.set_title(r'\textbf{(c)}', position=(-0.4,0.85), fontsize=font_size_2)
ax.set_xlim([-0.16, 0.05])
ax.set_ylim([-0.71, 0.1])
ax.plot(posl[:,0], posl[:,1],color='b')
posl_ = posl[::10]
D_traj_ = D_traj[::10]
for i in range(len(posl_)):
    plot_ellipse(ax, posl_[i], D_traj_[i]*1e-5,'c')

ax = fig.add_subplot(1, 3, 2)
ax.set_xlabel(r'$s_1$',fontsize=font_size_2)
ax.set_ylabel(r'$s_2$',fontsize=font_size_2,rotation=0)
ax.set_title(r'\textbf{(b)}', position=(-0.4,0.85), fontsize=font_size_2)
ax.set_xlim([-0.16, 0.05])
ax.set_ylim([-0.71, 0.1])
ax.plot(posl[:,0], posl[:,1],color='b')
posl_ = posl[::10]
S_traj_ = S_traj[::10]
for i in range(len(posl_)):
    plot_ellipse(ax, posl_[i], S_traj_[i]*1e-5,'m')
plt.subplots_adjust(left=0.09, bottom=0.25, right=0.99, top=0.95, wspace=0.55, hspace=0)
plt.show(block=True)
fig.savefig("pos2_energy.pdf")


