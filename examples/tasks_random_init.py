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

font_size_1 = 24
font_size_2 = 28
plt.rcParams.update({'font.size': font_size_1})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# plt.rc('lines', linewidth=10)
############################################
'''
pos2 k8 selected.
k0 2, k1 3, k2 3, k4 4, k8 2, k16 4

'''
############################################

base_np_filename = '/home/shahbaz/Software/garage/examples/np/data/local'
# prefix = 'blocks-random-init-pos'
prefix = 'blocks-random-init-pos2'
exp_name_0 = 'itr0'
exp_name_4 = 'itr4'
exp_name_9 = 'itr9'
exp_name_19 = 'itr19'
exp_name_39 = 'itr39'
exp_name_49 = 'itr49'
# exp_name_49 = 'itr30'
dir_name_0 = base_np_filename + '/' + prefix + '/' + exp_name_0
dir_name_4 = base_np_filename + '/' + prefix + '/' + exp_name_4
dir_name_9 = base_np_filename + '/' + prefix + '/' + exp_name_9
dir_name_19 = base_np_filename + '/' + prefix + '/' + exp_name_19
dir_name_39 = base_np_filename + '/' + prefix + '/' + exp_name_39
dir_name_49 = base_np_filename + '/' + prefix + '/' + exp_name_49

K = 8
exp_name = '2'
plt.rcParams["figure.figsize"] = (6,6)

filename = dir_name_0 + '/' + 'exp_log.pkl'
infile = open(filename, 'rb')
exp_log_0 = pickle.load(infile)
infile.close()
filename = dir_name_4 + '/' + 'exp_log.pkl'
infile = open(filename, 'rb')
exp_log_4 = pickle.load(infile)
infile.close()
filename = dir_name_9 + '/' + 'exp_log.pkl'
infile = open(filename, 'rb')
exp_log_9 = pickle.load(infile)
infile.close()
filename = dir_name_19 + '/' + 'exp_log.pkl'
infile = open(filename, 'rb')
exp_log_19 = pickle.load(infile)
infile.close()
filename = dir_name_39 + '/' + 'exp_log.pkl'
infile = open(filename, 'rb')
exp_log_39 = pickle.load(infile)
infile.close()
filename = dir_name_49 + '/' + 'exp_log.pkl'
infile = open(filename, 'rb')
exp_log_49 = pickle.load(infile)
infile.close()

filename = dir_name_0 + '/' + 'exp_param.pkl'
infile = open(filename, 'rb')
exp_param_0 = pickle.load(infile)
infile.close()
filename = dir_name_4 + '/' + 'exp_param.pkl'
infile = open(filename, 'rb')
exp_param_4 = pickle.load(infile)
infile.close()
filename = dir_name_9 + '/' + 'exp_param.pkl'
infile = open(filename, 'rb')
exp_param_9 = pickle.load(infile)
infile.close()
filename = dir_name_19 + '/' + 'exp_param.pkl'
infile = open(filename, 'rb')
exp_param_19 = pickle.load(infile)
infile.close()
filename = dir_name_39 + '/' + 'exp_param.pkl'
infile = open(filename, 'rb')
exp_param_39 = pickle.load(infile)
infile.close()
filename = dir_name_49 + '/' + 'exp_param.pkl'
infile = open(filename, 'rb')
exp_param_49 = pickle.load(infile)
infile.close()

U = 30
V = 30
s1 = np.linspace(-0.5, 0.5, U)
s2 = np.linspace(-1., 0.3, V)
S1, S2 = np.meshgrid(s1, s2)
label_pad = 20
title_x = 0.5
title_y = 1.1
SUCCESS_DIST = 0.025

fig = plt.figure()
epoch = exp_log_0[0]
obs0 = epoch[0]['observations']
pos0 = obs0[:,:2] - GOAL
T = pos0.shape[0]
dt = 0.01
tm = np.array(range(T))*dt
vel0 = np.zeros((T,2))
param_ep = exp_param_0[0]
param0 = param_ep['epoc_params'][0]
assert(K == len(param0[2]))

F = np.zeros((U,V))
for i in range(U):
    for j in range(V):
        S = np.array([S1[i,j], S2[i,j]])
        S_dot = np.zeros(2)
        F[i,j] = iMOGIC_energy_blocks(S, S_dot, param0, K, M=2)
# plt.axis('off')

# plt.legend(loc='upper right', bbox_to_anchor=(2., 1.),frameon=False,ncol=3)
ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax.contour3D(S1, S2, F, 30, label='Energy function', colors='g', alpha=0.7)
# ax.plot_wireframe(S1, S2, F, alpha=.7,label='Energy function')
ax.plot_surface(S1, S2, F,
                cmap=cm.coolwarm, edgecolor='none',alpha=0.7)
ax.set_xlabel(r'$s_1$',fontsize=font_size_2,labelpad=label_pad)
ax.set_ylabel(r'$s_2$',fontsize=font_size_2,labelpad=label_pad)
ax.set_zlabel(r'$V_{\theta|\dot{s}=0}$',fontsize=font_size_2,labelpad=label_pad)
ax.set_title(r'\textbf{(a)}\ Iteration\ 0', position=(title_x,title_y), fontsize=font_size_2)
ax.scatter(0,0,0,color='r',marker='o',s=100)
plt.subplots_adjust(left=0.0, bottom=0.1, right=0.85, top=0.9, wspace=0.2, hspace=0.2)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
Ns = 5
Ns = 5
mask = [True, True, True, True, True]
succ_count = 0
for i in range(Ns):
    obs = epoch[i]['observations']
    pos = obs[:, :2] - GOAL
    if np.min(np.linalg.norm(pos, axis=1)) < SUCCESS_DIST:
        succ_count = succ_count+1
    v = iMOGIC_energy_block_vec(pos, vel0, param0, K, M=2)
    if mask[i]:
        ax.plot3D(pos[:, 0], pos[:, 1], v, color='b', linewidth=5)
print('itr0', succ_count)
# for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
#     axis.line.set_linewidth(3)
# v0 = np.zeros(v0.shape)
##########################itr0##########################

fig = plt.figure()
epoch = exp_log_9[0]
obs0 = epoch[0]['observations']
pos0 = obs0[:,:2] - GOAL
T = pos0.shape[0]
dt = 0.01
tm = np.array(range(T))*dt
vel0 = np.zeros((T,2))
param_ep = exp_param_9[0]
param0 = param_ep['epoc_params'][0]
assert(K == len(param0[2]))

F = np.zeros((U,V))
for i in range(U):
    for j in range(V):
        S = np.array([S1[i,j], S2[i,j]])
        S_dot = np.zeros(2)
        F[i,j] = iMOGIC_energy_blocks(S, S_dot, param0, K, M=2)
# plt.axis('off')

# plt.legend(loc='upper right', bbox_to_anchor=(2., 1.),frameon=False,ncol=3)
ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax.contour3D(S1, S2, F, 30, label='Energy function', colors='g', alpha=0.7)
# ax.plot_wireframe(S1, S2, F, alpha=.7,label='Energy function')
ax.plot_surface(S1, S2, F,
                cmap=cm.coolwarm, edgecolor='none',alpha=0.7)
ax.set_xlabel(r'$s_1$',fontsize=font_size_2,labelpad=label_pad)
ax.set_ylabel(r'$s_2$',fontsize=font_size_2,labelpad=label_pad)
ax.set_zlabel(r'$V_{\theta|\dot{s}=0}$',fontsize=font_size_2,labelpad=label_pad)
ax.set_title(r'\textbf{(c)}\ Iteration\ 9', position=(title_x,title_y), fontsize=font_size_2)
ax.scatter(0,0,0,color='r',marker='o',s=100)
plt.subplots_adjust(left=0.0, bottom=0.1, right=0.85, top=0.9, wspace=0.2, hspace=0.2)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
Ns = 5
Ns = 5
succ_count = 0
mask = [True, True, True, True, True]
for i in range(Ns):
    obs = epoch[i]['observations']
    pos = obs[:, :2] - GOAL
    if np.min(np.linalg.norm(pos, axis=1)) < SUCCESS_DIST:
        succ_count = succ_count+1
    v = iMOGIC_energy_block_vec(pos, vel0, param0, K, M=2)
    if mask[i]:
        ax.plot3D(pos[:, 0], pos[:, 1], v, color='b', linewidth=5)
print('itr9', succ_count)
# for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
#     axis.line.set_linewidth(3)
# v0 = np.zeros(v0.shape)
##########################itr9##########################

fig = plt.figure()
epoch = exp_log_19[0]
obs0 = epoch[0]['observations']
pos0 = obs0[:,:2] - GOAL
T = pos0.shape[0]
dt = 0.01
tm = np.array(range(T))*dt
vel0 = np.zeros((T,2))
param_ep = exp_param_19[0]
param0 = param_ep['epoc_params'][0]
assert(K == len(param0[2]))

F = np.zeros((U,V))
for i in range(U):
    for j in range(V):
        S = np.array([S1[i,j], S2[i,j]])
        S_dot = np.zeros(2)
        F[i,j] = iMOGIC_energy_blocks(S, S_dot, param0, K, M=2)
# plt.axis('off')

# plt.legend(loc='upper right', bbox_to_anchor=(2., 1.),frameon=True,ncol=3)
ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax.contour3D(S1, S2, F, 30, label='Energy function', colors='g', alpha=0.7)
# ax.plot_wireframe(S1, S2, F, alpha=.7,label='Energy function')
ax.plot_surface(S1, S2, F,
                cmap=cm.coolwarm, edgecolor='none',alpha=0.7)
ax.set_xlabel(r'$s_1$',fontsize=font_size_2,labelpad=label_pad)
ax.set_ylabel(r'$s_2$',fontsize=font_size_2,labelpad=label_pad)
ax.set_zlabel(r'$V_{\theta|\dot{s}=0}$',fontsize=font_size_2,labelpad=label_pad)
ax.set_title(r'\textbf{(d)}\ Iteration\ 19', position=(title_x,title_y), fontsize=font_size_2)
ax.scatter(0,0,0,color='r',marker='o',s=100)
plt.subplots_adjust(left=0.0, bottom=0.1, right=0.85, top=0.9, wspace=0.2, hspace=0.2)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
Ns = 5
Ns = 5
succ_count = 0
mask = [True, True, True, True, True]
for i in range(Ns):
    obs = epoch[i]['observations']
    pos = obs[:, :2] - GOAL
    if np.min(np.linalg.norm(pos, axis=1)) < SUCCESS_DIST:
        succ_count = succ_count+1
    v = iMOGIC_energy_block_vec(pos, vel0, param0, K, M=2)
    if mask[i]:
        ax.plot3D(pos[:, 0], pos[:, 1], v, color='b', linewidth=5)
print('itr19', succ_count)
# for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
#     axis.line.set_linewidth(3)
# v0 = np.zeros(v0.shape)
##########################itr19##########################

fig = plt.figure()
epoch = exp_log_4[0]
obs0 = epoch[0]['observations']
pos0 = obs0[:,:2] - GOAL
T = pos0.shape[0]
dt = 0.01
tm = np.array(range(T))*dt
vel0 = np.zeros((T,2))
param_ep = exp_param_4[0]
param0 = param_ep['epoc_params'][0]
assert(K == len(param0[2]))

F = np.zeros((U,V))
for i in range(U):
    for j in range(V):
        S = np.array([S1[i,j], S2[i,j]])
        S_dot = np.zeros(2)
        F[i,j] = iMOGIC_energy_blocks(S, S_dot, param0, K, M=2)
# plt.axis('off')

# plt.legend(loc='upper right', bbox_to_anchor=(2., 1.),frameon=False,ncol=3)
ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax.contour3D(S1, S2, F, 30, label='Energy function', colors='g', alpha=0.7)
# ax.plot_wireframe(S1, S2, F, alpha=.7,label='Energy function')
ax.plot_surface(S1, S2, F,
                cmap=cm.coolwarm, edgecolor='none',alpha=0.7)
ax.set_xlabel(r'$s_1$',fontsize=font_size_2,labelpad=label_pad)
ax.set_ylabel(r'$s_2$',fontsize=font_size_2,labelpad=label_pad)
ax.set_zlabel(r'$V_{\theta|\dot{s}=0}$',fontsize=font_size_2,labelpad=label_pad)
ax.set_title(r'\textbf{(b)}\ Iteration\ 4', position=(title_x,title_y), fontsize=font_size_2)
ax.scatter(0,0,0,color='r',marker='o',s=100)
plt.subplots_adjust(left=0.0, bottom=0.1, right=0.85, top=0.9, wspace=0.2, hspace=0.2)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
Ns = 5
Ns = 5
succ_count = 0
mask = [True, True, True, True, True]
for i in range(Ns):
    obs = epoch[i]['observations']
    pos = obs[:, :2] - GOAL
    if np.min(np.linalg.norm(pos, axis=1)) < SUCCESS_DIST:
        succ_count = succ_count+1
    v = iMOGIC_energy_block_vec(pos, vel0, param0, K, M=2)
    if mask[i]:
        ax.plot3D(pos[:, 0], pos[:, 1], v, color='b', linewidth=5)
print('itr4', succ_count)
# for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
#     axis.line.set_linewidth(3)
# v0 = np.zeros(v0.shape)
##########################itr4#########################

fig = plt.figure()
# plt.axis('off')
# ax.grid(b=None)
# ax.set_xticks([])
epoch = exp_log_49[0]
s1 = np.linspace(-0.3, 0.3, U)
s2 = np.linspace(-1., 0.3, V)
S1, S2 = np.meshgrid(s1, s2)
param_ep = exp_param_49[0]
paraml = param_ep['epoc_params'][0]
F = np.zeros((U,V))
for i in range(U):
    for j in range(V):
        S = np.array([S1[i,j], S2[i,j]])
        S_dot = np.zeros(2)
        F[i,j] = iMOGIC_energy_blocks(S, S_dot, paraml, K, M=2)

ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax.plot_wireframe(S1, S2, F, alpha=.7,label='Energy function')
# ax.contour3D(S1, S2, F, 30, label='Energy function', colors='g', alpha=0.7)
ax.plot_surface(S1, S2, F,
                cmap=cm.coolwarm, edgecolor='none',alpha=0.7,linewidth=2)
Ns = 5
succ_count = 0
mask = [True, True, True, True, True]
for i in range(Ns):
    obs = epoch[i]['observations']
    pos = obs[:, :2] - GOAL
    if np.min(np.linalg.norm(pos, axis=1)) < SUCCESS_DIST:
        succ_count = succ_count+1
    v = iMOGIC_energy_block_vec(pos, vel0, paraml, K, M=2)
    if mask[i]:
        ax.plot3D(pos[:,0], pos[:,1], v, color='b', linewidth=5)
print('itr49', succ_count)

ax.scatter(0,0,0,color='r',marker='o',s=100)
ax.set_xlabel(r'$s_1$',fontsize=font_size_2,labelpad=label_pad)
ax.set_ylabel(r'$s_2$',fontsize=font_size_2,labelpad=label_pad)
ax.set_zlabel(r'$V_{\theta|\dot{s}=0}$',fontsize=font_size_2,labelpad=label_pad)
ax.set_title(r'\textbf{(e)}\ Iteration\ 49', position=(title_x,title_y), fontsize=font_size_2)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
# for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
#     axis.line.set_linewidth(3)
# ax.legend()


plt.subplots_adjust(left=0.0, bottom=0.1, right=0.85, top=0.9, wspace=0.2, hspace=0.2)
# plt.legend(loc='upper left', bbox_to_anchor=(-.5, 1.02),frameon=False,ncol=3)
# ax.set_title(r'\textbf{(e)}', position=(0.1,0.9), fontsize=font_size_2)
# plt.axis('off')
# ax.grid(b=None)

plt.show(block=True)




