# input: nz, nx, dz, dx, nSteps, nPoints_pml, nPad, dt, f0, survey_fname, data_dir_name, scratch_dir_name, isAc
using JSON
using DataStructures
using Dierckx
using Statistics
using PyCall
using LinearAlgebra
np = pyimport("numpy")

function paraGen(nz, nx, dz, dx, nSteps, dt, f0, nPml, nPad, para_fname, survey_fname, data_dir_name; if_win=false, filter_para=nothing, if_src_update=false, scratch_dir_name::String="", if_cross_misfit=false)

  para = OrderedDict()
  para["nz"] = nz
	para["nx"] = nx
	para["dz"] = dz
	para["dx"] = dx
	para["nSteps"] = nSteps
	para["dt"] = dt
	para["f0"] = f0
	para["nPoints_pml"] = nPml
	para["nPad"] = nPad
  if if_win != false
    para["if_win"] = true
  end
  if filter_para != nothing
    para["filter"] = filter_para
  end
  if if_src_update != false
    para["if_src_update"] = true
  end
	para["survey_fname"] = survey_fname
  para["data_dir_name"] = data_dir_name
  if !isdir(data_dir_name)
    mkdir(data_dir_name)
  end
  # if nStepsWrap != nothing
  #   para["nStepsWrap"] = nStepsWrap
  # end
  if if_cross_misfit != false
    para["if_cross_misfit"] = true
  end

	if(scratch_dir_name != "")
      para["scratch_dir_name"] = scratch_dir_name
      if !isdir(scratch_dir_name)
        mkdir(scratch_dir_name)
      end
  end
  para_string = JSON.json(para)

  open(para_fname,"w") do f
    write(f, para_string)
  end
end

# all shots share the same number of receivers
function surveyGen(z_src, x_src, z_rec, x_rec, survey_fname; Windows=nothing, 
  Weights=nothing, Src_Weights=nothing, Src_rxz=nothing, Rec_rxz=nothing)

  nsrc = length(x_src)
  nrec = length(x_rec)
  survey = OrderedDict()
  survey["nShots"] = nsrc
  for i = 1:nsrc
    shot = OrderedDict()
    shot["z_src"] = z_src[i]
    shot["x_src"] = x_src[i]
    shot["nrec"] = nrec
    shot["z_rec"] = z_rec
    shot["x_rec"] = x_rec
    if Windows != nothing
      shot["win_start"] = Windows["shot$(i-1)"][:start]
      shot["win_end"] = Windows["shot$(i-1)"][:end]
    end
    if Weights != nothing
      # shot["weights"] = Int64.(Weights["shot$(i-1)"][:weights])
      shot["weights"] = Weights["shot$(i-1)"][:weights]
    end
    if Src_Weights != nothing
      # shot["weights"] = Int64.(Weights["shot$(i-1)"][:weights])
      shot["src_weight"] = Src_Weights[i]
    end
    if Src_rxz != nothing
      shot["src_rxz"] = Src_rxz[i]
    end
    if Rec_rxz != nothing
      shot["rec_rxz"] = Rec_rxz
    end
    survey["shot$(i-1)"] = shot
  end
  
  survey_string = JSON.json(survey)
  open(survey_fname,"w") do f
    write(f, survey_string)
  end

end

function sourceGene(f, nStep, delta_t)
#  Ricker wavelet generation and integration for source
#  Dongzhuo Li @ Stanford
#  May, 2015

  e = pi*pi*f*f;
  t_delay = 1.2/f;
  source = Matrix{Float64}(undef, 1, nStep)
  for it = 1:nStep
      source[it] = (1-2*e*(delta_t*(it-1)-t_delay)^2)*exp(-e*(delta_t*(it-1)-t_delay)^2);
  end
  # return source

  for it = 2:nStep
      source[it] = source[it] + source[it-1];
  end
  source = source * delta_t;
end

# get vs high and low bounds from log point cloud 
# 1st row of Bounds: vp ref line
# 2nd row of Bounds: vs high ref line
# 3rd row of Bounds: vs low ref line
function cs_bounds_cloud(cpImg, Bounds)
  cs_high_itp = Spline1D(Bounds[1,:], Bounds[2,:]; k=1)
  cs_low_itp = Spline1D(Bounds[1,:], Bounds[3,:]; k=1)
  csHigh = zeros(size(cpImg))
  csLow = zeros(size(cpImg))
  for i = 1:size(cpImg, 1)
    for j = 1:size(cpImg, 2)
      csHigh[i,j] = min(cs_high_itp(cpImg[i,j]), cpImg[i,j]/sqrt(2)-1.0)
      csLow[i,j] = cs_low_itp(cpImg[i,j])
    end
  end
  return csHigh, csLow
end

function klauderWave(fmin, fmax, t_sweep, nStepTotal, nStepDelay, delta_t)
#  Klauder wavelet
#  Dongzhuo Li @ Stanford
#  August, 2019
  nStep = nStepTotal - nStepDelay
  source = Matrix{Float64}(undef, 1, nStep+nStep-1)
  source_half = Matrix{Float64}(undef, 1, nStep-1)
  K = (fmax - fmin) / t_sweep
  f0 = (fmin + fmax) / 2.0
  t_axis = delta_t:delta_t:(nStep-1)*delta_t
  source_half = sin.(pi * K .* t_axis .* (t_sweep .- t_axis)) .* cos.(2.0 * pi * f0 .* t_axis) ./ (pi*K.*t_axis*t_sweep)
  for i = 1:nStep-1
    source[i] = source_half[end-i+1]
  end
  for i = nStep+1:2*nStep-1
    source[i] = source_half[i-nStep]
  end
  source[nStep] = 1.0
  source_crop = source[:,nStep-nStepDelay:end]
  return source_crop
end

# function klauderWave(fmin, fmax, t_sweep, nStep, delta_t)
# #  Klauder wavelet
# #  Dongzhuo Li @ Stanford
# #  August, 2019
#   source = Matrix{Float64}(undef, 1, nStep)
#   K = (fmax - fmin) / t_sweep
#   f0 = (fmin + fmax) / 2.0
#   t_axis = delta_t:delta_t:(nStep-1)*delta_t
#   source_part = sin.(pi * K .* t_axis .* (t_sweep .- t_axis)) .* cos.(2.0 * pi * f0 .* t_axis) ./ (pi*K.*t_axis*t_sweep)
#   for i = 2:nStep
#     source[i] = source_part[i-1]
#   end
#   source[1] = 1.0
#   return source
# end

function computeRsxxzz(Vp, Vs, ind_z, ind_x)
  # since for c++ the index starts from 0
  ind_z .+= 1
  ind_x .+= 1
  Vp_pad = np.pad(Vp, ((4,4),(4,4)), "edge")
  Vs_pad = np.pad(Vs, ((4,4),(4,4)), "edge")
  Vp_pad
  ind_z .+= 4
  ind_x .+= 4
  rxz = zeros(Float64,size(ind_z))
  array_length = length(ind_z)
  Mask = ones(9,9)
  Mask[5,5] = 0.0
  for i = 1:array_length
    vp_ave = mean(Vp_pad[ind_z[i]-4:ind_z[i]+4, ind_x[i]-4:ind_x[i]+4].*Mask)
    vs_ave = mean(Vs_pad[ind_z[i]-4:ind_z[i]+4, ind_x[i]-4:ind_x[i]+4].*Mask)
    # rxz[i] = (vp_ave^2 - vs_ave^2)/(vp_ave^2 - 2*vs_ave^2) #for 2
    rxz[i] = (vp_ave^2)/(vp_ave^2 - 2*vs_ave^2) # for 3
  end
  return rxz
end

function weightObsTraces(obsPath, scratch, scaledPath, nShots, nJump, nt, nx)
  # obsPath: the path to observed data 
  # scaledPath: the path to which we save scaled data
  # scratch: where we can find preconditioned observed and synthetic data

  if !isdir(scaledPath)
    mkdir(scaledPath)
  end

  for i=1:nJump:nShots
    S_obs = read(obsPath * "/Shot$(i-1).bin")
    S_obs = reshape(reinterpret(Float32, S_obs), (nt, nx))

    S_condobs = read(scratch * "/CondObs_Shot$(i-1).bin")
    S_condobs = reshape(reinterpret(Float32, S_condobs), (nt, nx))

    S_syn = read(scratch * "/Syn_Shot$(i-1).bin")
    S_syn = reshape(reinterpret(Float32, S_syn), (nt, nx))

    S_fact_obs = zeros(Float32, size(S_obs))

    for iTrace = 1:nx
      fact_nomi = norm(S_syn[:, iTrace])
      fact_denomi = norm(S_condobs[:, iTrace])
      if fact_denomi != 0
        println("ratio = $(fact_nomi/(fact_denomi))")
        S_fact_obs[:, iTrace] = S_obs[:, iTrace] * fact_nomi/(fact_denomi+1e-12)
        # S_fact_obs[:, iTrace] = S_obs[:, iTrace]
      end
    end

    open(scaledPath*"/Shot$(i-1).bin", "w") do f
      write(f, S_fact_obs)
    end
  end

end
    


    