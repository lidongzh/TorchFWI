#include "Boundary.h"
#include "Parameter.h"
#include "utilities.h"

Bnd::Bnd(const Parameter &para) {
  withAdj_ = para.withAdj();
  if (withAdj_) {
    nz_ = para.nz();
    nx_ = para.nx();
    nPml_ = para.nPoints_pml();
    nPad_ = para.nPad();
    nSteps_ = para.nSteps();

    // nzBnd_ = nz_ - 2*nPml_ - nPad_;
    // nxBnd_ = nx_ - 2*nPml_;
    // save extra 2 layers in the pml for derivative at the boundaries
    nzBnd_ = nz_ - 2 * nPml_ - nPad_ + 4;
    nxBnd_ = nx_ - 2 * nPml_ + 4;
    nLayerStore_ = 5;

    // nzBnd_ = nz_ - nPad_; // DL save pml
    // nxBnd_ = nx_; // DL save pml
    // nLayerStore_ = nPml_+3; //

    // len_Bnd_vec_ = 2 * (nzBnd_ + nxBnd_);
    len_Bnd_vec_ =
        2 * (nLayerStore_ * nzBnd_ + nLayerStore_ * nxBnd_);  // store n layers

    // allocate the boundary vector in the device
    CHECK(cudaMalloc((void **)&d_Bnd_szz,
                     len_Bnd_vec_ * nSteps_ * sizeof(float)));

    CHECK(cudaMalloc((void **)&d_Bnd_sxz,
                     len_Bnd_vec_ * nSteps_ * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_Bnd_sxx,
                     len_Bnd_vec_ * nSteps_ * sizeof(float)));

    CHECK(
        cudaMalloc((void **)&d_Bnd_vz, len_Bnd_vec_ * nSteps_ * sizeof(float)));
    CHECK(
        cudaMalloc((void **)&d_Bnd_vx, len_Bnd_vec_ * nSteps_ * sizeof(float)));
  }
}

Bnd::~Bnd() {
  if (withAdj_) {
    CHECK(cudaFree(d_Bnd_szz));
    CHECK(cudaFree(d_Bnd_sxz));
    CHECK(cudaFree(d_Bnd_sxx));
    CHECK(cudaFree(d_Bnd_vz));
    CHECK(cudaFree(d_Bnd_vx));
  }
}

void Bnd::field_from_bnd(float *d_szz, float *d_sxz, float *d_sxx, float *d_vz,
                         float *d_vx, int indT) {
  from_bnd<<<(len_Bnd_vec_ + 31) / 32, 32>>>(d_szz, d_Bnd_szz, nz_, nx_, nzBnd_,
                                             nxBnd_, len_Bnd_vec_, nLayerStore_,
                                             indT, nPml_, nPad_, nSteps_);

  from_bnd<<<(len_Bnd_vec_ + 31) / 32, 32>>>(d_sxz, d_Bnd_sxz, nz_, nx_, nzBnd_,
                                             nxBnd_, len_Bnd_vec_, nLayerStore_,
                                             indT, nPml_, nPad_, nSteps_);

  from_bnd<<<(len_Bnd_vec_ + 31) / 32, 32>>>(d_sxx, d_Bnd_sxx, nz_, nx_, nzBnd_,
                                             nxBnd_, len_Bnd_vec_, nLayerStore_,
                                             indT, nPml_, nPad_, nSteps_);

  from_bnd<<<(len_Bnd_vec_ + 31) / 32, 32>>>(d_vz, d_Bnd_vz, nz_, nx_, nzBnd_,
                                             nxBnd_, len_Bnd_vec_, nLayerStore_,
                                             indT, nPml_, nPad_, nSteps_);

  from_bnd<<<(len_Bnd_vec_ + 31) / 32, 32>>>(d_vx, d_Bnd_vx, nz_, nx_, nzBnd_,
                                             nxBnd_, len_Bnd_vec_, nLayerStore_,
                                             indT, nPml_, nPad_, nSteps_);
}

void Bnd::field_to_bnd(float *d_szz, float *d_sxz, float *d_sxx, float *d_vz,
                       float *d_vx, int indT, bool if_stress) {
  if (if_stress) {
    to_bnd<<<(len_Bnd_vec_ + 31) / 32, 32>>>(d_szz, d_Bnd_szz, nz_, nx_, nzBnd_,
                                             nxBnd_, len_Bnd_vec_, nLayerStore_,
                                             indT, nPml_, nPad_, nSteps_);

    to_bnd<<<(len_Bnd_vec_ + 31) / 32, 32>>>(d_sxz, d_Bnd_sxz, nz_, nx_, nzBnd_,
                                             nxBnd_, len_Bnd_vec_, nLayerStore_,
                                             indT, nPml_, nPad_, nSteps_);

    to_bnd<<<(len_Bnd_vec_ + 31) / 32, 32>>>(d_sxx, d_Bnd_sxx, nz_, nx_, nzBnd_,
                                             nxBnd_, len_Bnd_vec_, nLayerStore_,
                                             indT, nPml_, nPad_, nSteps_);

  } else {
    to_bnd<<<(len_Bnd_vec_ + 31) / 32, 32>>>(d_vz, d_Bnd_vz, nz_, nx_, nzBnd_,
                                             nxBnd_, len_Bnd_vec_, nLayerStore_,
                                             indT, nPml_, nPad_, nSteps_);

    to_bnd<<<(len_Bnd_vec_ + 31) / 32, 32>>>(d_vx, d_Bnd_vx, nz_, nx_, nzBnd_,
                                             nxBnd_, len_Bnd_vec_, nLayerStore_,
                                             indT, nPml_, nPad_, nSteps_);
  }
}