import torch
import torch.nn as nn
import numpy as np 
from torch.utils.cpp_extension import load
path = '../Ops/FWI/Src'
import matplotlib.pyplot as plt
import os
from scipy import optimize
import fwi_utils as ft
from collections import OrderedDict

os.makedirs(path+'/build/', exist_ok=True)
def load_fwi(path):
    fwi = load(name="fwi",
            sources=[path+'/Torch_Fwi.cpp', path+'/Parameter.cpp', path+'/libCUFD.cu', path+'/el_stress.cu', path+'/el_velocity.cu', path+'/el_stress_adj.cu', path+'/el_velocity_adj.cu', path+'/Model.cu', path+'/Cpml.cu', path+'/utilities.cu',	path+'/Src_Rec.cu', path+'/Boundary.cu'],
            extra_cflags=[
                '-O3 -fopenmp -lpthread'
            ],
            extra_include_paths=['/usr/local/cuda/include', path+'/rapidjson'],
            extra_ldflags=['-L/usr/local/cuda/lib64 -lnvrtc -lcuda -lcudart -lcufft'],
            build_directory=path+'/build/',
            verbose=True)
    return fwi

fwi_ops = load_fwi(path)

# class FWIFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, Lambda, Mu, Den, Stf, gpu_id, Shot_ids, para_fname):
#         misfit, = fwi_ops.forward(Lambda, Mu, Den, Stf, gpu_id, Shot_ids, para_fname)
#         variables = [Lambda, Mu, Den, Stf]
#         ctx.save_for_backward(*variables)
#         ctx.gpu_id = gpu_id
#         ctx.Shot_ids = Shot_ids
#         ctx.para_fname = para_fname
#         return misfit

#     @staticmethod
#     def backward(ctx, grad_misfit):
#         outputs = fwi_ops.backward(*ctx.saved_variables, ctx.gpu_id, ctx.Shot_ids, ctx.para_fname)
#         grad_Lambda, grad_Mu, grad_Den, grad_stf = outputs
#         return grad_Lambda, grad_Mu, grad_Den, grad_stf, None, None, None

class FWIFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Lambda, Mu, Den, Stf, ngpu, Shot_ids, para_fname):
        outputs = fwi_ops.backward(Lambda, Mu, Den, Stf, ngpu, Shot_ids, para_fname)
        ctx.outputs = outputs[1:]
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_misfit):
        grad_Lambda, grad_Mu, grad_Den, grad_stf = ctx.outputs
        return grad_Lambda, grad_Mu, grad_Den, grad_stf, None, None, None

class FWI(torch.nn.Module):
    def __init__(self, Vp, Vs, Den, Stf, opt, Mask=None, Vp_bounds=None, \
        Vs_bounds=None, Den_bounds=None):
        super(FWI, self).__init__()

        self.nz = opt['nz']
        self.nx = opt['nx']
        self.nz_orig = opt['nz_orig']
        self.nx_orig = opt['nx_orig']
        self.nPml = opt['nPml']
        self.nPad = opt['nPad']

        self.Bounds = {}
        if Vp.requires_grad:
            self.Vp = nn.Parameter(Vp)
            if Vp_bounds != None:
                self.Bounds['Vp'] = Vp_bounds
        else:
            self.Vp = Vp
        if Vs.requires_grad:
            self.Vs = nn.Parameter(Vs)
            if Vs_bounds != None:
                self.Bounds['Vs'] = Vs_bounds
        else:
            self.Vs = Vs
        if Den.requires_grad:
            self.Den = nn.Parameter(Den)
            if Den_bounds != None:
                self.Bounds['Den'] = Den_bounds
        else:
            self.Den = Den

        if Mask == None:
            self.Mask = torch.ones((self.nz+2*self.nPml+self.nPad, \
                self.nx+2*self.nPml), dtype=torch.float32)
        else:
            self.Mask = Mask

        self.Stf = Stf
        self.para_fname = opt['para_fname']

    def forward(self, Shot_ids, ngpu=1):
        Vp_pad, Vs_pad, Den_pad = ft.padding(self.Vp, self.Vs, self.Den,\
            self.nz_orig, self.nx_orig, self.nz, self.nx, self.nPml, self.nPad)
        Vp_ref = Vp_pad.clone().detach()
        Vs_ref = Vs_pad.clone().detach()
        Den_ref = Den_pad.clone().detach()
        Vp_mask_pad = self.Mask * Vp_pad + (1.0 - self.Mask) * Vp_ref
        Vs_mask_pad = self.Mask * Vs_pad + (1.0 - self.Mask) * Vs_ref
        Den_mask_pad = self.Mask * Den_pad + (1.0 - self.Mask) * Den_ref
        Lambda = (Vp_mask_pad**2 - 2.0 * Vs_mask_pad**2) * Den_mask_pad / 1e6
        Mu = Vs_mask_pad**2 * Den_mask_pad / 1e6
        return FWIFunction.apply(Lambda, Mu, Den_mask_pad, self.Stf, ngpu, Shot_ids, self.para_fname)

class FWI_obscalc(torch.nn.Module):
    def __init__(self, Vp, Vs, Den, Stf, para_fname):
        super(FWI_obscalc, self).__init__()
        self.Lambda = (Vp**2 - 2.0 * Vs**2) * Den / 1e6
        self.Mu = Vs**2 * Den / 1e6
        self.Den = Den
        self.Stf = Stf
        self.para_fname = para_fname

    def forward(self, Shot_ids, ngpu=1):
        fwi_ops.obscalc(self.Lambda, self.Mu, self.Den, self.Stf, ngpu, Shot_ids, self.para_fname)
