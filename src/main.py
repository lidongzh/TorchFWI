import torch
import numpy as np
import scipy.io as sio
import sys
import os
sys.path.append("../Ops/FWI")
from FWI_ops import *
import matplotlib.pyplot as plt
import fwi_utils as ft
import argparse
from scipy import optimize
from obj_wrapper import PyTorchObjective

# get parameters from command line
parser = argparse.ArgumentParser()
parser.add_argument('--generate_data', action='store_true')
parser.add_argument('--exp_name', type=str, default='default')
parser.add_argument('--nIter', type=int, default=100)
parser.add_argument('--ngpu', type=int, default=1)
args = vars(parser.parse_args())
generate_data = args['generate_data']
exp_name = args['exp_name']
nIter = args['nIter']
ngpu = args['ngpu']

# ========== parameters ============
oz = 0.0 # original depth
ox = 0.0
dz_orig = 24.0 # original scale
dx_orig = 24.0 # original scale
nz_orig = 134 # original scale
nx_orig = 384 # original scale
dz = dz_orig/1.0
dx = dx_orig/1.0
nz = round((dz_orig * nz_orig) / dz)
nx = round((dx_orig * nx_orig) / dx)
dt = 0.0025
nSteps = 2000
nPml = 32
nPad = int(32 - np.mod((nz+2*nPml), 32))
nz_pad = nz + 2*nPml + nPad
nx_pad = nx + 2*nPml

Mask = np.zeros((nz_pad, nx_pad))
Mask[nPml:nPml+nz, nPml:nPml+nx] = 1.0
Mask[nPml:nPml+10,:] = 0.0
th_mask = torch.tensor(Mask, dtype=torch.float32)

filter = [[0.0, 0.0, 2.0, 2.5],
         [0.0, 0.0, 2.0, 3.5],
         [0.0, 0.0, 2.0, 4.5],
         [0.0, 0.0, 2.0, 5.5],
         [0.0, 0.0, 2.0, 6.5],
         [0.0, 0.0, 2.0, 7.5]]

f0_vec = [4.5]
if_src_update = False
if_win = False

ind_src_x = np.arange(4, 385, 8).astype(int)
ind_src_z = 2*np.ones(ind_src_x.shape[0]).astype(int)
ind_rec_x = np.arange(3, 382).astype(int)
ind_rec_z = 2*np.ones(ind_rec_x.shape[0]).astype(int)

para_fname = './' + exp_name + '/para_file.json'
survey_fname = './' + exp_name + '/survey_file.json'
data_dir_name = './' + exp_name + '/Data'
ft.paraGen(nz_pad, nx_pad, dz, dx, nSteps, dt, f0_vec[0], nPml, nPad, \
    para_fname, survey_fname, data_dir_name)
ft.surveyGen(ind_src_z, ind_src_x, ind_rec_z, ind_rec_x, survey_fname)

# Stf = ft.sourceGene(f0_vec[0], nSteps, dt)
Stf = sio.loadmat("../Mar_models/sourceF_4p5_2_high.mat", \
  squeeze_me=True, struct_as_record=False)["sourceF"]
th_Stf = torch.tensor(Stf, dtype=torch.float32, \
  requires_grad=False).repeat(len(ind_src_x), 1)

Shot_ids = torch.tensor(np.arange(0,48), dtype=torch.int32)
# Shot_ids = torch.tensor([24], dtype=torch.int32)
# ==========

if (generate_data == True):
  cp_true_pad = np.fromfile('../Mar_models/Model_Cp_true.bin', dtype='float32', count=-1)
  cp_true_pad = np.ascontiguousarray(np.reshape(cp_true_pad, (nz_pad, -1), order='F'))
  cs_true_pad = np.zeros((nz_pad, nx_pad))
  print(f'cp_true_pad shape = {cp_true_pad.shape}')
  # plt.imshow(cp_true_pad, cmap='RdBu_r')
  # plt.colorbar()
  # plt.show()
  den_true_pad = 2500.0 * np.ones((nz_pad, nx_pad))
  th_cp_pad = torch.tensor(cp_true_pad, dtype=torch.float32, requires_grad=False)
  th_cs_pad = torch.tensor(cs_true_pad, dtype=torch.float32, requires_grad=False)
  th_den_pad = torch.tensor(den_true_pad, dtype=torch.float32, requires_grad=False)
  
  fwi_obscalc = FWI_obscalc(th_cp_pad, th_cs_pad, th_den_pad, th_Stf, para_fname)
  fwi_obscalc(Shot_ids, ngpu=ngpu)
  sys.exit('End of Data Generation')

########## Inversion ###########
opt = {}
opt['nz'] = nz
opt['nx'] = nx 
opt['nz_orig'] = nz_orig
opt['nx_orig'] = nx_orig
opt['nPml'] = nPml
opt['nPad'] = nPad
opt['para_fname'] = para_fname
cp_init_pad = np.fromfile('../Mar_models/Model_Cp_init_1D.bin', dtype='float32', count=-1)
cp_init_pad = np.ascontiguousarray(np.reshape(cp_init_pad, (nz_pad, -1), order='F'))
# plt.imshow(cp_init_pad, cmap='RdBu_r');
# plt.colorbar()
# plt.show()
cs_init_pad = np.zeros((nz, nx))
den_init_pad = 2500.0 * np.ones((nz, nx))
th_cp_inv = torch.tensor(cp_init_pad[nPml:nPml+nz, nPml:nPml+nx], dtype=torch.float32, requires_grad=True)
th_cs_inv = torch.tensor(cs_init_pad, dtype=torch.float32, requires_grad=False)
th_den_inv = torch.tensor(den_init_pad, dtype=torch.float32, requires_grad=False)

# Vp_bounds = [1500.0 * np.ones((nz, nx)), 5500.0 * np.ones((nz, nx))]
Vp_bounds = None

fwi = FWI(th_cp_inv, th_cs_inv, th_den_inv, th_Stf, opt, Mask=th_mask, Vp_bounds=Vp_bounds, \
        Vs_bounds=None, Den_bounds=None)

compLoss = lambda : fwi(Shot_ids, ngpu=ngpu)
obj = PyTorchObjective(fwi, compLoss)


__iter = 0
result_dir_name = './' + exp_name + '/Results'
def save_prog(x):
  global __iter
  os.makedirs(result_dir_name, exist_ok=True)
  with open(result_dir_name + '/loss.txt', 'a') as text_file:
    text_file.write("%d %s\n" % (__iter, obj.f))
  sio.savemat(result_dir_name + '/cp' + str(__iter) + '.mat', \
    {'cp':fwi.Vp.cpu().detach().numpy()})
  sio.savemat(result_dir_name + '/grad_cp' + str(__iter) + \
    '.mat', {'grad_cp':fwi.Vp.grad.cpu().detach().numpy()})
  __iter = __iter + 1

maxiter = nIter
optimize.minimize(obj.fun, obj.x0, method='L-BFGS-B', jac=obj.jac, bounds=obj.bounds, \
  tol=None, callback=save_prog, options={'disp': True, 'iprint': 101, \
  'gtol': 1e-012, 'maxiter': maxiter, 'ftol': 1e-12, 'maxcor': 30, 'maxfun': 15000})

ImgInv = fwi.Vp.cpu().detach().numpy()
plt.rcParams.update({'font.size': 12})
plt.figure(0)
plt.imshow(ImgInv, cmap='RdBu_r', \
  extent=[0,(nx_orig-1)*dx_orig,0,(nz_orig-1)*dz_orig])
plt.xlabel('x (m)')
plt.ylabel('z (m)')
cb = plt.colorbar()
cb.set_label("Vp (m/s)")
plt.show()