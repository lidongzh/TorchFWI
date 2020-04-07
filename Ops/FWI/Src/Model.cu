#include <iostream>
#include <string>
#include "Model.h"
#include "Parameter.h"
#include "utilities.h"

// model default constructor
Model::Model() {
  std::cout << "ERROR: You need to supply parameters to initialize models!"
            << std::endl;
  exit(1);
}

// model constructor from parameter file
Model::Model(const Parameter &para, const float *Lambda_, const float *Mu_,
             const float *Den_) {
  nz_ = para.nz();
  nx_ = para.nx();

  dim3 threads(32, 16);
  dim3 blocks((nz_ + 32 - 1) / 32, (nx_ + 16 - 1) / 16);

  h_Lambda = (float *)malloc(nz_ * nx_ * sizeof(float));
  h_Mu = (float *)malloc(nz_ * nx_ * sizeof(float));
  h_Den = (float *)malloc(nz_ * nx_ * sizeof(float));
  h_Cp = (float *)malloc(nz_ * nx_ * sizeof(float));
  h_Cs = (float *)malloc(nz_ * nx_ * sizeof(float));
  h_CpGrad = (float *)malloc(nz_ * nx_ * sizeof(float));
  h_LambdaGrad = (float *)malloc(nz_ * nx_ * sizeof(float));
  h_MuGrad = (float *)malloc(nz_ * nx_ * sizeof(float));
  h_DenGrad = (float *)malloc(nz_ * nx_ * sizeof(float));
  h_StfGrad = (float *)malloc(para.nSteps() * sizeof(float));

  // load Vp, Vs, and Den binaries

  for (int i = 0; i < nz_ * nx_; i++) {
    if (Lambda_[i] < 0.0) {
      printf("Lambda is negative!!!");
      // exit(1);
    }
    h_Lambda[i] = Lambda_[i];
    h_Mu[i] = Mu_[i];
    h_Den[i] = Den_[i];
  }

  initialArray(h_Cp, nz_ * nx_, 0.0);
  initialArray(h_Cs, nz_ * nx_, 0.0);
  initialArray(h_CpGrad, nz_ * nx_, 0.0);
  initialArray(h_LambdaGrad, nz_ * nx_, 0.0);
  initialArray(h_MuGrad, nz_ * nx_, 0.0);
  initialArray(h_DenGrad, nz_ * nx_, 0.0);
  initialArray(h_StfGrad, para.nSteps(), 0.0);

  CHECK(cudaMalloc((void **)&d_Lambda, nz_ * nx_ * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_Mu, nz_ * nx_ * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_Den, nz_ * nx_ * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_Cp, nz_ * nx_ * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_Cs, nz_ * nx_ * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_ave_Mu, nz_ * nx_ * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_ave_Byc_a, nz_ * nx_ * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_ave_Byc_b, nz_ * nx_ * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_CpGrad, nz_ * nx_ * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_LambdaGrad, nz_ * nx_ * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_MuGrad, nz_ * nx_ * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_DenGrad, nz_ * nx_ * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_StfGrad, para.nSteps() * sizeof(float)));
  intialArrayGPU<<<blocks, threads>>>(d_ave_Mu, nz_, nx_, 0.0);
  intialArrayGPU<<<blocks, threads>>>(d_CpGrad, nz_, nx_, 0.0);
  intialArrayGPU<<<blocks, threads>>>(d_LambdaGrad, nz_, nx_, 0.0);
  intialArrayGPU<<<blocks, threads>>>(d_MuGrad, nz_, nx_, 0.0);
  intialArrayGPU<<<blocks, threads>>>(d_DenGrad, nz_, nx_, 0.0);
  intialArrayGPU<<<blocks, threads>>>(d_ave_Byc_a, nz_, nx_, 1.0 / 1000.0);
  intialArrayGPU<<<blocks, threads>>>(d_ave_Byc_b, nz_, nx_, 1.0 / 1000.0);
  intialArrayGPU<<<blocks, threads>>>(d_StfGrad, para.nSteps(), 1, 0.0);

  CHECK(cudaMemcpy(d_Lambda, h_Lambda, nz_ * nx_ * sizeof(float),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_Mu, h_Mu, nz_ * nx_ * sizeof(float),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_Den, h_Den, nz_ * nx_ * sizeof(float),
                   cudaMemcpyHostToDevice));

  // moduliInit<<<blocks, threads>>>(d_Cp, d_Cs, d_Den, d_Lambda, d_Mu, nz_,
  // nx_);
  velInit<<<blocks, threads>>>(d_Lambda, d_Mu, d_Den, d_Cp, d_Cs, nz_, nx_);
  aveMuInit<<<blocks, threads>>>(d_Mu, d_ave_Mu, nz_, nx_);
  aveBycInit<<<blocks, threads>>>(d_Den, d_ave_Byc_a, d_ave_Byc_b, nz_, nx_);

  CHECK(cudaMemcpy(h_Cp, d_Cp, nz_ * nx_ * sizeof(float),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h_Cs, d_Cs, nz_ * nx_ * sizeof(float),
                   cudaMemcpyDeviceToHost));
}

Model::~Model() {
  free(h_Cp);
  free(h_Cs);
  free(h_Den);
  free(h_Lambda);
  free(h_Mu);
  free(h_CpGrad);
  CHECK(cudaFree(d_Cp));
  CHECK(cudaFree(d_Cs));
  CHECK(cudaFree(d_Den));
  CHECK(cudaFree(d_Lambda));
  CHECK(cudaFree(d_Mu));
  CHECK(cudaFree(d_ave_Mu));
  CHECK(cudaFree(d_ave_Byc_a));
  CHECK(cudaFree(d_ave_Byc_b));
  CHECK(cudaFree(d_CpGrad));
  CHECK(cudaFree(d_LambdaGrad));
  CHECK(cudaFree(d_MuGrad));
  CHECK(cudaFree(d_DenGrad));
  CHECK(cudaFree(d_StfGrad));
}