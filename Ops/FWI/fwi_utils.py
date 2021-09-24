import torch
import torch.nn.functional as F
from collections import OrderedDict
import json
import os
import numpy as np

def padding(cp, cs, den, nz_orig, nx_orig, nz, nx, nPml, nPad):
    tran_cp = cp.view(1, 1, nz_orig, nx_orig)
    tran_cs = cs.view(1, 1, nz_orig, nx_orig)
    tran_den = den.view(1, 1, nz_orig, nx_orig)
    tran_cp2 = F.interpolate(tran_cp, size=(nz,nx), mode='bilinear', align_corners=False)
    tran_cs2 = F.interpolate(tran_cs, size=(nz,nx), mode='bilinear', align_corners=False)
    tran_den2 = F.interpolate(tran_den, size=(nz,nx), mode='bilinear', align_corners=False)
    tran_cp3 = F.pad(tran_cp2, pad=(nPml, nPml, nPml, (nPml+nPad)), mode='replicate')
    tran_cs3 = F.pad(tran_cs2, pad=(nPml, nPml, nPml, (nPml+nPad)), mode='replicate')
    tran_den3 = F.pad(tran_den2, pad=(nPml, nPml, nPml, (nPml+nPad)), mode='replicate')
    cp_pad = tran_cp3.view(nz+2*nPml+nPad, nx+2*nPml)
    cs_pad = tran_cs3.view(nz+2*nPml+nPad, nx+2*nPml)
    den_pad = tran_den3.view(nz+2*nPml+nPad, nx+2*nPml)
    return cp_pad, cs_pad, den_pad

def paraGen(nz, nx, dz, dx, nSteps, dt, f0, nPml, nPad, para_fname, survey_fname, \
    data_dir_name, if_win=False, filter_para=None, if_src_update=False, \
    scratch_dir_name='', if_cross_misfit=False):
    # para = OrderedDict()
    para = {}
    para['nz'] = nz
    para['nx'] = nx
    para['dz'] = dz
    para['dx'] = dx
    para['nSteps'] = nSteps
    para['dt'] = dt
    para['f0'] = f0
    para['nPoints_pml'] = nPml
    para['nPad'] = nPad

    if if_win != False:
        para['if_win'] = True

    if filter_para != None:
        para['filter'] = filter_para

    if if_src_update != False:
        para['if_src_update'] = True
    
    para['survey_fname'] = survey_fname
    para['data_dir_name'] = data_dir_name

    os.makedirs(data_dir_name, exist_ok=True)
 
    if if_cross_misfit != False:
        para['if_cross_misfit'] = True
    
    if(scratch_dir_name != ''):
        para['scratch_dir_name'] = scratch_dir_name
        os.makedirs(scratch_dir_name, exist_ok=True)

    with open(para_fname, 'w') as fp:
        json.dump(para, fp)


# all shots share the same number of receivers
def surveyGen(z_src, x_src, z_rec, x_rec, survey_fname, Windows=None, \
    Weights=None, Src_Weights=None, Src_rxz=None, Rec_rxz=None):
    x_src = x_src.tolist()
    z_src = z_src.tolist()
    x_rec = x_rec.tolist()
    z_rec = z_rec.tolist()
    nsrc = len(x_src)
    nrec = len(x_rec)
    survey = {}
    survey['nShots'] = nsrc
    for i in range(0, nsrc):
        shot = {}
        shot['z_src'] = z_src[i]
        shot['x_src'] = x_src[i]
        shot['nrec'] = nrec
        shot['z_rec'] = z_rec
        shot['x_rec'] = x_rec
        if Windows != None:
            shot['win_start'] = Windows['shot' + str(i)][:start]
            shot['win_end'] = Windows['shot' + str(i)][:end]
    
        if Weights != None:
            shot['weights'] = Weights['shot' + str(i)][:weights]
            
        if Src_Weights != None:
            shot['src_weight'] = Src_Weights[i]
            
        if Src_rxz != None:
            shot['src_rxz'] = Src_rxz[i]
            
        if Rec_rxz != None:
            Rec_rxz.tolist()
            shot['rec_rxz'] = Rec_rxz
        
        survey['shot' + str(i)] = shot
    
    with open(survey_fname, 'w') as fp:
        json.dump(survey, fp)


def sourceGene(f, nStep, delta_t):
#  Ricker wavelet generation and integration for source
#  Dongzhuo Li @ Stanford
#  May, 2015

  e = np.pi * np.pi * f * f
  t_delay = 1.2/f
  source = np.zeros((nStep))
  for it in range(0,nStep):
      source[it] = (1-2*e*(delta_t*(it)-t_delay)**2)*np.exp(-e*(delta_t*(it)-t_delay)**2)

  # return source

  for it in range(1,nStep):
      source[it] = source[it] + source[it-1]

  source = source * delta_t

  return source
  
