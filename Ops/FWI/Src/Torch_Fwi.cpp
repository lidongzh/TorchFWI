#include <stdio.h>
#include <torch/extension.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include "libCUFD.h"
#include "omp.h"
// using namespace std;

std::vector<torch::Tensor> fwi_forward(const torch::Tensor &th_Lambda,
                                       const torch::Tensor &th_Mu,
                                       const torch::Tensor &th_Den,
                                       const torch::Tensor &th_stf, int gpu_id,
                                       const torch::Tensor &th_shot_ids,
                                       const string para_fname) {
  float *misfit_ptr = nullptr;
  misfit_ptr = (float *)malloc(sizeof(float));
  // transform from torch tensor to 1D native array
  auto Lambda = th_Lambda.data_ptr<float>();
  auto Mu = th_Mu.data_ptr<float>();
  auto Den = th_Den.data_ptr<float>();
  auto stf = th_stf.data_ptr<float>();
  auto shot_ids = th_shot_ids.data_ptr<int>();
  const int group_size = th_shot_ids.size(0);
  cufd(misfit_ptr, nullptr, nullptr, nullptr, nullptr, Lambda, Mu, Den, stf, 0, gpu_id,
       group_size, shot_ids, para_fname);
  torch::Tensor th_misfit = torch::from_blob(misfit_ptr, {1});
  return {th_misfit.clone()};
}

std::vector<torch::Tensor> fwi_backward(const torch::Tensor &th_Lambda,
                                        const torch::Tensor &th_Mu,
                                        const torch::Tensor &th_Den,
                                        const torch::Tensor &th_stf, int ngpu,
                                        const torch::Tensor &th_shot_ids,
                                        const string para_fname) {
  const int nz = th_Lambda.size(0);
  const int nx = th_Lambda.size(1);
  const int nSrc = th_stf.size(0);
  const int nSteps = th_stf.size(1);
  const int group_size = th_shot_ids.size(0);
  if (ngpu > group_size) {
    printf("The number of GPUs should be smaller than the number of shots!\n");
    exit(1);
  }
  // transform from torch tensor to 1D native array
  auto Lambda = th_Lambda.data_ptr<float>();
  auto Mu = th_Mu.data_ptr<float>();
  auto Den = th_Den.data_ptr<float>();
  auto stf = th_stf.data_ptr<float>();
  auto sepBars = torch::linspace(0, group_size, ngpu + 1,
                                 torch::TensorOptions().dtype(torch::kFloat32));
  std::vector<float> vec_misfit(ngpu);
  std::vector<torch::Tensor> vec_grad_Lambda(ngpu);
  std::vector<torch::Tensor> vec_grad_Mu(ngpu);
  std::vector<torch::Tensor> vec_grad_Den(ngpu);
  std::vector<torch::Tensor> vec_grad_stf(ngpu);
  auto th_grad_Lambda_sum = torch::zeros_like(th_Lambda);
  auto th_grad_Mu_sum = torch::zeros_like(th_Mu);
  auto th_grad_Den_sum = torch::zeros_like(th_Den);
  float misfit_sum = 0.0;

#pragma omp parallel for num_threads(ngpu)
  for (int i = 0; i < ngpu; i++) {
    // float *misfit_ptr = nullptr;
    // misfit_ptr = (float *)malloc(sizeof(float));
    float misfit = 0.0;
    auto th_grad_Lambda = torch::zeros_like(th_Lambda);
    auto th_grad_Mu = torch::zeros_like(th_Mu);
    auto th_grad_Den = torch::zeros_like(th_Den);
    auto th_grad_stf = torch::zeros_like(th_stf);
    int startBar = round(sepBars[i].item<int>());
    int endBar = round(sepBars[i + 1].item<int>());
    auto th_sub_shot_ids = th_shot_ids.narrow(0, startBar, endBar - startBar);
    auto shot_ids = th_sub_shot_ids.data_ptr<int>();
    cufd(&misfit, th_grad_Lambda.data_ptr<float>(),
         th_grad_Mu.data_ptr<float>(), th_grad_Den.data_ptr<float>(),
         th_grad_stf.data_ptr<float>(), Lambda, Mu, Den, stf, 1, i,
         th_sub_shot_ids.size(0), shot_ids, para_fname);
    vec_grad_Lambda.at(i) = th_grad_Lambda;
    vec_grad_Mu.at(i) = th_grad_Mu;
    vec_grad_Den.at(i) = th_grad_Den;
    vec_grad_stf.at(i) = th_grad_stf;
    // torch::Tensor th_misfit = torch::from_blob(&misfit, {1});
    vec_misfit.at(i) = misfit;
  }
  for (int i = 0; i < ngpu; i++) {
    th_grad_Lambda_sum += vec_grad_Lambda.at(i);
    th_grad_Mu_sum += vec_grad_Mu.at(i);
    th_grad_Den_sum += vec_grad_Den.at(i);
    misfit_sum += vec_misfit.at(i);
  }
  return {torch::tensor({misfit_sum}), th_grad_Lambda_sum, th_grad_Mu_sum,
          th_grad_Den_sum, vec_grad_stf.at(0)};
}

void fwi_obscalc(const torch::Tensor &th_Lambda, const torch::Tensor &th_Mu,
                 const torch::Tensor &th_Den, const torch::Tensor &th_stf, int ngpu,
                 const torch::Tensor &th_shot_ids, const string para_fname) {
  const int group_size = th_shot_ids.size(0);
  if (ngpu > group_size) {
    printf("The number of GPUs should be smaller than the number of shots!\n");
    exit(1);
  }
  // transform from torch tensor to 1D native array
  auto Lambda = th_Lambda.data_ptr<float>();
  auto Mu = th_Mu.data_ptr<float>();
  auto Den = th_Den.data_ptr<float>();
  auto stf = th_stf.data_ptr<float>();
  auto sepBars = torch::linspace(0, group_size, ngpu + 1,
                                 torch::TensorOptions().dtype(torch::kFloat32));
#pragma omp parallel for num_threads(ngpu)
  for (int i = 0; i < ngpu; i++) {
    int startBar = round(sepBars[i].item<int>());
    int endBar = round(sepBars[i + 1].item<int>());
    auto th_sub_shot_ids = th_shot_ids.narrow(0, startBar, endBar - startBar);
    auto shot_ids = th_sub_shot_ids.data_ptr<int>();
    cufd(nullptr, nullptr, nullptr, nullptr, nullptr, Lambda, Mu, Den, stf, 2, i,
         th_sub_shot_ids.size(0), shot_ids, para_fname);
    //     cufd(nullptr, nullptr, nullptr, nullptr, nullptr, Lambda, Mu, Den, stf, 2, gpu_id,
    //          group_size, shot_ids, para_fname);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fwi_forward, "forward");
  m.def("backward", &fwi_backward, "backward");
  m.def("obscalc", &fwi_obscalc, "obscalc");
}