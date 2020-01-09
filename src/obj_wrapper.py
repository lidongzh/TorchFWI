import torch
import torch.nn.functional as F
import math
import numpy as np
from scipy import optimize
from functools import reduce
from collections import OrderedDict
import matplotlib.pyplot as plt

class PyTorchObjective(object):
    """PyTorch objective function, wrapped to be called by scipy.optimize. 
    Modified from https://gist.github.com/gngdb/a9f912df362a85b37c730154ef3c294b"""
    def __init__(self, obj, loss):
        self.obj = obj # some pytorch module, that produces a scalar loss
        self.loss = loss
        # make an x0 from the parameters in this module
        parameters = OrderedDict(obj.named_parameters())
        self.param_shapes = {n:parameters[n].size() for n in parameters}
        # ravel and concatenate all parameters to make x0
        self.x0 = np.concatenate([parameters[n].data.cpu().numpy().ravel() 
                                   for n in parameters]).astype(np.float64)
        if self.obj.Bounds != {}:
            self.bounds = self.pack_bounds()
        else:
            self.bounds = None

    def unpack_parameters(self, x):
        """optimize.minimize will supply 1D array, chop it up for each parameter."""
        i = 0
        named_parameters = OrderedDict()
        for n in self.param_shapes:
            param_len = reduce(lambda x,y: x*y, self.param_shapes[n])
            # slice out a section of this length
            param = x[i:i+param_len]
            # reshape according to this size, and cast to torch
            param = param.reshape(*self.param_shapes[n])
            named_parameters[n] = torch.from_numpy(param)
            # update index
            i += param_len
        return named_parameters

    def pack_grads(self):
        """pack all the gradients from the parameters in the module into a
        numpy array."""
        grads = []
        for p in self.obj.parameters():
            grad = p.grad.data.cpu().numpy()
            grads.append(grad.ravel())
        return np.concatenate(grads).astype(np.float64)
    
    def pack_bounds(self):
        """set up bounds for L-BFGS-B"""
        lbounds = []
        ubounds = []
        for n in self.param_shapes:
            lbound, ubound = self.obj.Bounds[n][0], self.obj.Bounds[n][1]
            lbounds.append(lbound.ravel())
            ubounds.append(ubound.ravel())
        return optimize.Bounds(np.concatenate(lbounds).astype(np.float64), \
            np.concatenate(ubounds).astype(np.float64))

    def is_new(self, x):
        # if this is the first thing we've seen
        if not hasattr(self, 'cached_x'):
            return True
        else:
            # compare x to cached_x to determine if we've been given a new input
            x, self.cached_x = np.array(x), np.array(self.cached_x)
            error = np.abs(x - self.cached_x)
            return error.max() > 1e-8

    def cache(self, x):
        # unpack x and load into module 
        state_dict = self.unpack_parameters(x)
        for name,buf in self.obj.named_buffers():
            state_dict[name] = buf
        self.obj.load_state_dict(state_dict)
        self.cached_x = x
        # zero the gradient
        self.obj.zero_grad()
        obj = self.loss()
        self.f = obj.item()
        obj.backward()
        self.jac = self.pack_grads()


    def fun(self, x):
        if self.is_new(x):
            self.cache(x)
        return self.f

    def jac(self, x):
        if self.is_new(x):
            self.cache(x)
        # plt.imshow(self.jac.reshape(134, 384))
        # plt.colorbar()
        # plt.show()
        return self.jac