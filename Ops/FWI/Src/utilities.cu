#define d_Mu(x, y) d_Mu[(y) * (nx) + (x)]
#define d_ave_Mu(x, y) d_ave_Mu[(y) * (nx) + (x)]
#define d_field(z, x) d_field[(x) * (nz) + (z)]
#define d_bnd(x, indT) d_bnd[(indT) * (len_Bnd_vec) + (x)]
#define d_Den(x, y) d_Den[(y) * (nx) + (x)]
#define d_ave_Byc_a(x, y) d_ave_Byc_a[(y) * (nx) + (x)]
#define d_ave_Byc_b(x, y) d_ave_Byc_b[(y) * (nx) + (x)]
#include "utilities.h"

void fileBinLoad(float *h_bin, int size, std::string fname) {
  FILE *fp = fopen(fname.c_str(), "rb");
  if (fp == nullptr) {
    std::cout << "Attempted to read " << fname << std::endl;
    printf("File reading error!\n");
    exit(1);
  } else {
    size_t sizeRead = fread(h_bin, sizeof(float), size, fp);
  }
  fclose(fp);
}

void fileBinWrite(float *h_bin, int size, std::string fname) {
  FILE *fp = fopen(fname.c_str(), "wb");
  if (fp == nullptr) {
    printf("File writing error!\n");
    exit(1);
  } else {
    fwrite(h_bin, sizeof(float), size, fp);
  }
  fclose(fp);
}

void fileBinWriteDouble(double *h_bin, int size, std::string fname) {
  FILE *fp = fopen(fname.c_str(), "wb");
  if (fp == nullptr) {
    printf("File writing error!\n");
    exit(1);
  } else {
    fwrite(h_bin, sizeof(double), size, fp);
  }
  fclose(fp);
}

void initialArray(float *ip, int size, float value) {
  for (int i = 0; i < size; i++) {
    ip[i] = value;
    // printf("value = %f\n", value);
  }
}

void initialArray(double *ip, int size, double value) {
  for (int i = 0; i < size; i++) {
    ip[i] = value;
    // printf("value = %f\n", value);
  }
}

__global__ void intialArrayGPU(float *ip, int nx, int ny, float value) {
  int gidx = threadIdx.x + blockDim.x * blockIdx.x;
  int gidy = threadIdx.y + blockDim.y * blockIdx.y;
  if (gidx < nx && gidy < ny) {
    int offset = gidx + gidy * nx;
    ip[offset] = value;
  }
}

__global__ void assignArrayGPU(float *ip_in, float *ip_out, int nx, int ny) {
  int gidx = threadIdx.x + blockDim.x * blockIdx.x;
  int gidy = threadIdx.y + blockDim.y * blockIdx.y;
  if (gidx < nx && gidy < ny) {
    int offset = gidx + gidy * nx;
    ip_out[offset] = ip_in[offset];
  }
}

void displayArray(std::string s, float *ip, int nx, int ny) {
  // printf("ip: \n");
  // printf("%s: \n", s);
  std::cout << s << ": " << std::endl;
  for (int i = 0; i < ny; i++) {
    for (int j = 0; j < nx; j++) {
      // printf("ip[%d, %d] = %f  ", i, j, ip[i*nx+j]);
      printf("%f  ", ip[i * nx + j]);
    }
    printf("\n");
  }
  printf("\n\n\n");
}

__global__ void moduliInit(float *d_Cp, float *d_Cs, float *d_Den,
                           float *d_Lambda, float *d_Mu, int nx, int ny) {
  // printf("Hello, world!\n");
  int gidx = threadIdx.x + blockDim.x * blockIdx.x;
  int gidy = threadIdx.y + blockDim.y * blockIdx.y;
  int offset = gidx + gidy * nx;
  if (gidx < nx && gidy < ny) {
    // printf("offset = %d ", offset);
    // printf("gridDim.x = %d ", gridDim.x);
    // printf("blockIdx.y = %d ", blockIdx.y);
    d_Mu[offset] = powf(d_Cs[offset], 2) * d_Den[offset];
    d_Lambda[offset] =
        d_Den[offset] * (powf(d_Cp[offset], 2) - 2 * powf(d_Cs[offset], 2));
    if (d_Lambda[offset] < 0) {
      printf("Lambda is negative!!!");
    }
  }
}

__global__ void velInit(float *d_Lambda, float *d_Mu, float *d_Den, float *d_Cp,
                        float *d_Cs, int nx, int ny) {
  // printf("Hello, world!\n");
  int gidx = threadIdx.x + blockDim.x * blockIdx.x;
  int gidy = threadIdx.y + blockDim.y * blockIdx.y;
  int offset = gidx + gidy * nx;
  if (gidx < nx && gidy < ny) {
    // printf("offset = %d ", offset);
    // printf("gridDim.x = %d ", gridDim.x);
    // printf("blockIdx.y = %d ", blockIdx.y);
    d_Cp[offset] =
        sqrt((d_Lambda[offset] + 2.0 * d_Mu[offset]) / d_Den[offset]);
    d_Cs[offset] = sqrt((d_Mu[offset]) / d_Den[offset]);
  }
}

__global__ void aveMuInit(float *d_Mu, float *d_ave_Mu, int nx, int ny) {
  int gidx = threadIdx.x + blockDim.x * blockIdx.x;
  int gidy = threadIdx.y + blockDim.y * blockIdx.y;
  float a, b, c, d;
  if (gidx >= 2 && gidx <= nx - 3 && gidy >= 2 && gidy <= ny - 3) {
    a = d_Mu(gidx, gidy);
    b = d_Mu(gidx + 1, gidy);
    c = d_Mu(gidx, gidy + 1);
    d = d_Mu(gidx + 1, gidy + 1);
    if (a == 0.0 || b == 0.0 || c == 0.0 || d == 0.0) {
      d_ave_Mu(gidx, gidy) = 0.0;
    } else {
      d_ave_Mu(gidx, gidy) = 4.0 / (1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d);
    }
  }
}

__global__ void aveBycInit(float *d_Den, float *d_ave_Byc_a, float *d_ave_Byc_b,
                           int nx, int ny) {
  int gidx = threadIdx.x + blockDim.x * blockIdx.x;
  int gidy = threadIdx.y + blockDim.y * blockIdx.y;
  if (gidx >= 2 && gidx <= nx - 3 && gidy >= 2 && gidy <= ny - 3) {
    d_ave_Byc_a(gidx, gidy) = 2.0 / (d_Den(gidx + 1, gidy) + d_Den(gidx, gidy));
    d_ave_Byc_b(gidx, gidy) = 2.0 / (d_Den(gidx, gidy + 1) + d_Den(gidx, gidy));
  } else {
    return;
  }
}

__global__ void gpuMinus(float *d_out, float *d_in1, float *d_in2, int nx,
                         int ny) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int idy = threadIdx.y + blockDim.y * blockIdx.y;
  // only compute last N-1 time samples for misfits!!!!!!!! DL 02/25/2019
  if (idx < nx && idy < ny && idx > 0) {
    d_out[(idy) * (nx) + (idx)] =
        d_in1[(idy) * (nx) + (idx)] - d_in2[(idy) * (nx) + (idx)];
  } else if (idx == 0 && idy < ny) {
    d_out[(idy) * (nx) + (idx)] = 0.0;
  } else {
    return;
  }
}

__global__ void cuda_cal_objective(float *obj, float *err, int ng)
/*< calculate the value of objective function: obj >*/
{
  const int Block_Size = 512;
  __shared__ float sdata[Block_Size];
  int tid = threadIdx.x;
  sdata[tid] = 0.0f;
  for (int s = 0; s < (ng + Block_Size - 1) / Block_Size; s++) {
    int id = s * blockDim.x + threadIdx.x;
    float a = (id < ng) ? err[id] : 0.0f;
    // sdata[tid] += a*a;
    sdata[tid] += powf(a, 2);
  }
  __syncthreads();

  /* do reduction in shared mem */
  // for(int s=blockDim.x/2; s>32; s>>=1) {
  //     if (threadIdx.x < s) sdata[tid] += sdata[tid + s];
  //     __syncthreads();
  // }
  // if (tid < 32) {
  //     if (blockDim.x >=  64) { sdata[tid] += sdata[tid + 32]; }
  //     if (blockDim.x >=  32) { sdata[tid] += sdata[tid + 16]; }
  //     if (blockDim.x >=  16) { sdata[tid] += sdata[tid +  8]; }
  //     if (blockDim.x >=   8) { sdata[tid] += sdata[tid +  4]; }
  //     if (blockDim.x >=   4) { sdata[tid] += sdata[tid +  2]; }
  //     if (blockDim.x >=   2) { sdata[tid] += sdata[tid +  1]; }
  // }
  for (int s = blockDim.x / 2; s >= 1; s /= 2) {
    if (threadIdx.x < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  if (tid == 0) {
    *obj = sdata[0];
  }
}

float cal_objective(float *array, int N) {
  float misfit = 0.0;
  printf("hhh\n");
  for (int i = 0; i < N; i++) {
    misfit += array[i] * array[i];
  }
  return misfit;
}

float compCpAve(float *array, int N) {
  float temp = 0.0;
  for (int i = 0; i < N; i++) {
    temp += array[i];
  }
  temp = temp / float(N);
  return temp;
}

void compCourantNumber(float *h_Cp, int size, float dt, float dz, float dx) {
  float max = h_Cp[0];
  float Courant_number = 0.0;
  for (int i = 0; i < size; i++) {
    if (h_Cp[i] > max) {
      max = h_Cp[i];
    }
  }
  float dh_min = (dz < dx) ? dz : dx;
  // Courant_number = max * dt * sqrtf(powf(1.0 / dz, 2) + powf(1.0 / dx, 2));
  Courant_number = max * dt * sqrtf(2.0) * (1.0 / 24.0 + 9.0 / 8.0) / dh_min;
  
  if (Courant_number > 1.0) {
    std::cout << "Courant_number = " << Courant_number << std::endl;
    exit(1);
  }
}

void cpmlInit(float *K, float *a, float *b, float *K_half, float *a_half,
              float *b_half, int N, int nPml, float dh, float f0, float dt,
              float CpAve) {
  float *damp, *damp_half, *alpha, *alpha_half;
  float d0_h = 0.0;
  float Rcoef = 0.0008;
  float depth_in_pml = 0.0;
  float depth_normalized = 0.0;
  float thickness_PML = 0.0;
  // const float PI = 3.141592653589793238462643383279502884197169;
  const float K_MAX_PML = 2.0;
  const float ALPHA_MAX_PML = 2.0 * PI * (f0 / 2.0);
  const float NPOWER = 8.0;
  const float c1 = 0.25, c2 = 0.75, c3 = 0.0;
  // const float c1 = 0.0, c2 = 1.0, c3 = 0.0;

  thickness_PML = nPml * dh;  // changed here
  CpAve = 3000.0;             // DL make this model independent
  d0_h = -(NPOWER + 1) * CpAve * log(Rcoef) / (2.0 * thickness_PML);
  damp = (float *)malloc(N * sizeof(float));
  damp_half = (float *)malloc(N * sizeof(float));
  alpha = (float *)malloc(N * sizeof(float));
  alpha_half = (float *)malloc(N * sizeof(float));
  initialArray(damp, N, 0.0);
  initialArray(damp_half, N, 0.0);
  initialArray(K, N, 1.0);
  initialArray(K_half, N, 1.0);
  initialArray(alpha, N, 0.0);
  initialArray(alpha_half, N, 0.0);
  initialArray(a, N, 0.0);
  initialArray(a_half, N, 0.0);
  initialArray(b, N, 0.0);
  initialArray(b_half, N, 0.0);

  for (int i = 0; i < N; i++) {
    // left edge
    depth_in_pml = (nPml - i) * dh;
    if (depth_in_pml >= 0.0) {
      depth_normalized = depth_in_pml / thickness_PML;
      damp[i] =
          d0_h * (c1 * depth_normalized + c2 * pow(depth_normalized, NPOWER) +
                  c3 * pow(depth_normalized, 2 * NPOWER));
      K[i] = 1.0 + (K_MAX_PML - 1.0) * pow(depth_normalized, NPOWER);
      alpha[i] = ALPHA_MAX_PML * (1.0 - depth_normalized);
    }
    if (alpha[i] < 0.0) {
      std::cout << "CPML alpha < 0.0 --" << __LINE__ << std::endl;
      exit(1);
    }

    // half the grid points
    depth_in_pml = (nPml - i - 0.5) * dh;
    if (depth_in_pml >= 0.0) {
      depth_normalized = depth_in_pml / thickness_PML;
      damp_half[i] =
          d0_h * (c1 * depth_normalized + c2 * pow(depth_normalized, NPOWER) +
                  c3 * pow(depth_normalized, 2 * NPOWER));
      K_half[i] = 1.0 + (K_MAX_PML - 1.0) * pow(depth_normalized, NPOWER);
      alpha_half[i] = ALPHA_MAX_PML * (1.0 - depth_normalized);
    }
    if (alpha_half[i] < 0.0) {
      std::cout << "CPML alpha_half < 0.0 --" << __LINE__ << std::endl;
      exit(1);
    }

    // right edge
    depth_in_pml = (nPml - N + i) * dh;
    if (depth_in_pml >= 0.0) {
      depth_normalized = depth_in_pml / thickness_PML;
      damp[i] =
          d0_h * (c1 * depth_normalized + c2 * pow(depth_normalized, NPOWER) +
                  c3 * pow(depth_normalized, 2 * NPOWER));
      K[i] = 1.0 + (K_MAX_PML - 1.0) * pow(depth_normalized, NPOWER);
      alpha[i] = ALPHA_MAX_PML * (1.0 - depth_normalized);
    }
    if (alpha[i] < 0.0) {
      std::cout << "CPML alpha < 0.0 --" << __LINE__ << std::endl;
      exit(1);
    }

    depth_in_pml = (nPml - N + i + 0.5) * dh;
    if (depth_in_pml >= 0.0) {
      depth_normalized = depth_in_pml / thickness_PML;
      damp_half[i] =
          d0_h * (c1 * depth_normalized + c2 * pow(depth_normalized, NPOWER) +
                  c3 * pow(depth_normalized, 2 * NPOWER));
      K_half[i] = 1.0 + (K_MAX_PML - 1.0) * powf(depth_normalized, NPOWER);
      alpha_half[i] = ALPHA_MAX_PML * (1.0 - depth_normalized);
    }
    if (alpha_half[i] < 0.0) {
      std::cout << "CPML alpha_half < 0.0 --" << __LINE__ << std::endl;
      exit(1);
    }

    if (alpha[i] < 0.0) {
      alpha[i] = 0.0;
    }
    if (alpha_half[i] < 0.0) {
      alpha_half[i] = 0.0;
    }

    b[i] = expf(-(damp[i] / K[i] + alpha[i]) * dt);
    b_half[i] = expf(-(damp_half[i] / K_half[i] + alpha_half[i]) * dt);

    if (fabs(damp[i]) > 1.0e-6) {
      a[i] = damp[i] * (b[i] - 1.0) / (K[i] * (damp[i] + K[i] * alpha[i]));
    }
    if (fabs(damp_half[i]) > 1.0e-6) {
      a_half[i] = damp_half[i] * (b_half[i] - 1.0) /
                  (K_half[i] * (damp_half[i] + K_half[i] * alpha_half[i]));
    }
  }
  free(damp);
  free(damp_half);
  free(alpha);
  free(alpha_half);
}

// Dongzhuo Li 05/15/2019
__global__ void from_bnd(float *d_field, float *d_bnd, int nz, int nx,
                         int nzBnd, int nxBnd, int len_Bnd_vec, int nLayerStore,
                         int indT, int nPml, int nPad, int nSteps) {
  int idxBnd = threadIdx.x + blockDim.x * blockIdx.x;
  int iRow, jCol;

  if (idxBnd >= 0 && idxBnd <= nLayerStore * nzBnd - 1) {
    jCol = idxBnd / nzBnd;
    iRow = idxBnd - jCol * nzBnd;
    d_bnd(idxBnd, indT) = d_field((iRow + nPml - 2), (jCol + nPml - 2));
  } else if (idxBnd >= nLayerStore * nzBnd &&
             idxBnd <= 2 * nLayerStore * nzBnd - 1) {
    jCol = (idxBnd - nLayerStore * nzBnd) / nzBnd;
    iRow = (idxBnd - nLayerStore * nzBnd) - jCol * nzBnd;
    d_bnd(idxBnd, indT) =
        d_field((iRow + nPml - 2), (nx - nPml - jCol - 1 + 2));
  } else if (idxBnd >= 2 * nLayerStore * nzBnd &&
             idxBnd <= nLayerStore * (2 * nzBnd + nxBnd) - 1) {
    iRow = (idxBnd - 2 * nLayerStore * nzBnd) / nxBnd;
    jCol = (idxBnd - 2 * nLayerStore * nzBnd) - iRow * nxBnd;
    d_bnd(idxBnd, indT) = d_field((iRow + nPml - 2), (jCol + nPml - 2));
  } else if (idxBnd >= nLayerStore * (2 * nzBnd + nxBnd) &&
             idxBnd <= 2 * nLayerStore * (nzBnd + nxBnd) - 1) {
    iRow = (idxBnd - nLayerStore * (2 * nzBnd + nxBnd)) / nxBnd;
    jCol = (idxBnd - nLayerStore * (2 * nzBnd + nxBnd)) - iRow * nxBnd;
    d_bnd(idxBnd, indT) =
        d_field((nz - nPml - nPad - iRow - 1 + 2), (jCol + nPml - 2));
  } else {
    return;
  }
}

// Dongzhuo Li 05/15/2019
__global__ void to_bnd(float *d_field, float *d_bnd, int nz, int nx, int nzBnd,
                       int nxBnd, int len_Bnd_vec, int nLayerStore, int indT,
                       int nPml, int nPad, int nSteps) {
  int idxBnd = threadIdx.x + blockDim.x * blockIdx.x;
  int iRow, jCol;

  if (idxBnd >= 0 && idxBnd <= nLayerStore * nzBnd - 1) {
    jCol = idxBnd / nzBnd;
    iRow = idxBnd - jCol * nzBnd;
    d_field((iRow + nPml - 2), (jCol + nPml - 2)) = d_bnd(idxBnd, indT);
  } else if (idxBnd >= nLayerStore * nzBnd &&
             idxBnd <= 2 * nLayerStore * nzBnd - 1) {
    jCol = (idxBnd - nLayerStore * nzBnd) / nzBnd;
    iRow = (idxBnd - nLayerStore * nzBnd) - jCol * nzBnd;
    d_field((iRow + nPml - 2), (nx - nPml - jCol - 1 + 2)) =
        d_bnd(idxBnd, indT);
  } else if (idxBnd >= 2 * nLayerStore * nzBnd &&
             idxBnd <= nLayerStore * (2 * nzBnd + nxBnd) - 1) {
    iRow = (idxBnd - 2 * nLayerStore * nzBnd) / nxBnd;
    jCol = (idxBnd - 2 * nLayerStore * nzBnd) - iRow * nxBnd;
    d_field((iRow + nPml - 2), (jCol + nPml - 2)) = d_bnd(idxBnd, indT);
  } else if (idxBnd >= nLayerStore * (2 * nzBnd + nxBnd) &&
             idxBnd <= 2 * nLayerStore * (nzBnd + nxBnd) - 1) {
    iRow = (idxBnd - nLayerStore * (2 * nzBnd + nxBnd)) / nxBnd;
    jCol = (idxBnd - nLayerStore * (2 * nzBnd + nxBnd)) - iRow * nxBnd;
    d_field((nz - nPml - nPad - iRow - 1 + 2), (jCol + nPml - 2)) =
        d_bnd(idxBnd, indT);
  } else {
    return;
  }
}

// // Dongzhuo Li 02/24/2019
// __global__ void from_bnd(float *d_field, float *d_bnd, int nz, int nx, int
// nzBnd, \
//   int nxBnd, int len_Bnd_vec, int nLayerStore, int indT, int nPml, int nPad,
//   int nSteps) {

//     int idxBnd = threadIdx.x + blockDim.x*blockIdx.x;
//     int iRow,jCol;

//     if(idxBnd>=0 && idxBnd<=nLayerStore*nzBnd-1) {
//         jCol = idxBnd/nzBnd;
//         iRow = idxBnd - jCol*nzBnd;
//         d_bnd(idxBnd,indT) = d_field((iRow),(jCol));
//     }
//     else if(idxBnd>=nLayerStore*nzBnd && idxBnd<=2*nLayerStore*nzBnd-1){
//         jCol = (idxBnd-nLayerStore*nzBnd)/nzBnd;
//         iRow = (idxBnd-nLayerStore*nzBnd) - jCol*nzBnd;
//         d_bnd(idxBnd,indT) = d_field((iRow),(nx-jCol-1));
//     }
//     else if(idxBnd>=2*nLayerStore*nzBnd &&
//     idxBnd<=nLayerStore*(2*nzBnd+nxBnd)-1) {
//         iRow = (idxBnd - 2*nLayerStore*nzBnd)/nxBnd;
//         jCol = (idxBnd - 2*nLayerStore*nzBnd) - iRow*nxBnd;
//         d_bnd(idxBnd,indT) = d_field((iRow),(jCol));
//     }
//     else if(idxBnd>=nLayerStore*(2*nzBnd+nxBnd) &&
//     idxBnd<=2*nLayerStore*(nzBnd+nxBnd)-1) {
//         iRow = (idxBnd - nLayerStore*(2*nzBnd+nxBnd))/nxBnd;
//         jCol = (idxBnd - nLayerStore*(2*nzBnd+nxBnd)) - iRow*nxBnd;
//         d_bnd(idxBnd,indT) = d_field((nz-nPad-iRow-1),(jCol));
//     }
//     else {
//         return;
//     }

//     // if(idxBnd>=0 && idxBnd<=2*(nzBnd+nxBnd)-1) {
//     //  d_bnd(idxBnd, indT) = 1.0;
//     // } else {
//     //  return;
//     // }

// }

// // Dongzhuo Li 02/24/2019
// __global__ void to_bnd(float *d_field, float *d_bnd, int nz, int nx, int
// nzBnd, \
//   int nxBnd, int len_Bnd_vec, int nLayerStore, int indT, int nPml, int nPad,
//   int nSteps) {

//     int idxBnd = threadIdx.x + blockDim.x*blockIdx.x;
//     int iRow,jCol;

//     if(idxBnd>=0 && idxBnd<=nLayerStore*nzBnd-1) {
//         jCol = idxBnd/nzBnd;
//         iRow = idxBnd - jCol*nzBnd;
//         d_field((iRow),(jCol)) = d_bnd(idxBnd,indT);
//     }
//     else if(idxBnd>=nLayerStore*nzBnd && idxBnd<=2*nLayerStore*nzBnd-1){
//         jCol = (idxBnd-nLayerStore*nzBnd)/nzBnd;
//         iRow = (idxBnd-nLayerStore*nzBnd) - jCol*nzBnd;
//         d_field((iRow),(nx-jCol-1)) = d_bnd(idxBnd,indT);
//     }
//     else if(idxBnd>=2*nLayerStore*nzBnd &&
//     idxBnd<=nLayerStore*(2*nzBnd+nxBnd)-1) {
//         iRow = (idxBnd - 2*nLayerStore*nzBnd)/nxBnd;
//         jCol = (idxBnd - 2*nLayerStore*nzBnd) - iRow*nxBnd;
//         d_field((iRow),(jCol)) = d_bnd(idxBnd,indT);
//     }
//     else if(idxBnd>=nLayerStore*(2*nzBnd+nxBnd) &&
//     idxBnd<=2*nLayerStore*(nzBnd+nxBnd)-1) {
//         iRow = (idxBnd - nLayerStore*(2*nzBnd+nxBnd))/nxBnd;
//         jCol = (idxBnd - nLayerStore*(2*nzBnd+nxBnd)) - iRow*nxBnd;
//         d_field((nz-nPad-iRow-1),(jCol)) = d_bnd(idxBnd,indT);
//     }
//     else {
//         return;
//     }

// }

__global__ void src_rec_gauss_amp(float *gauss_amp, int nz, int nx) {
  int gidz = blockIdx.x * blockDim.x + threadIdx.x;
  int gidx = blockIdx.y * blockDim.y + threadIdx.y;
  if (gidz >= 0 && gidz < nz && gidx >= 0 && gidx < nx) {
    int idz = gidz - nz / 2;
    int idx = gidx - nx / 2;
    gauss_amp[gidz + gidx * nz] =
        expf(-1000.0 * (powf(float(idz), 2) + powf(float(idx), 2)));
    // printf("gidz=%d, gidx=%d, gauss_amp=%.10f\n", gidz, gidx,
    //        gauss_amp[gidz + gidx * nz]);
  } else {
    return;
  }
}

__global__ void add_source(float *d_szz, float *d_sxx, float amp, int nz,
                           bool isFor, int z_loc, int x_loc, float dt,
                           float *gauss_amp, double rxz) {
  // int id = threadIdx.x + blockDim.x * blockIdx.x;
  int gidz = blockIdx.x * blockDim.x + threadIdx.x;
  int gidx = blockIdx.y * blockDim.y + threadIdx.y;

  float scale = pow(1500.0, 2);
  if (isFor) {
    if (gidz >= 0 && gidz < 9 && gidx >= 0 && gidx < 9) {
      int idz = gidz - 9 / 2;
      int idx = gidx - 9 / 2;
      // printf("amp = %f  ", amp);
      d_szz[(z_loc + idz) + nz * (x_loc + idx)] +=
          scale * amp * dt * gauss_amp[gidz + gidx * 9];
      // crosswell borehole source (can be modified) assume cp/cs = sqrt(3.0)
      d_sxx[(z_loc + idz) + nz * (x_loc + idx)] +=
          rxz * scale * amp * dt * gauss_amp[gidz + gidx * 9];
    } else {
      return;
    }
  } else {
    if (gidz >= 0 && gidz < 9 && gidx >= 0 && gidx < 9) {
      int idz = gidz - 9 / 2;
      int idx = gidx - 9 / 2;
      // printf("amp = %f  ", amp);
      d_szz[(z_loc + idz) + nz * (x_loc + idx)] -=
          scale * amp * dt * gauss_amp[gidz + gidx * 9];
      d_sxx[(z_loc + idz) + nz * (x_loc + idx)] -=
          rxz * scale * amp * dt * gauss_amp[gidz + gidx * 9];
    } else {
      return;
    }
  }
}

__global__ void recording(float *d_szz, float *d_sxx, int nz, float *d_data,
                          int iShot, int it, int nSteps, int nrec, int *d_z_rec,
                          int *d_x_rec, double *d_rxz) {
  int iRec = threadIdx.x + blockDim.x * blockIdx.x;
  if (iRec >= nrec) {
    return;
  }
  d_data[(iRec) * (nSteps) + (it)] =
      d_szz[d_z_rec[iRec] + d_x_rec[iRec] * nz] +
      d_rxz[iRec] * d_sxx[d_z_rec[iRec] + d_x_rec[iRec] * nz];
}

__global__ void res_injection(float *d_szz_adj, float *d_sxx_adj, int nz,
                              float *d_res, int it, float dt, int nSteps,
                              int nrec, int *d_z_rec, int *d_x_rec,
                              double *d_rxz) {
  int iRec = threadIdx.x + blockDim.x * blockIdx.x;
  if (iRec >= nrec) {
    return;
  }
  d_szz_adj[d_z_rec[iRec] + nz * d_x_rec[iRec]] +=
      d_res[(iRec) * (nSteps) + (it)];
  d_sxx_adj[d_z_rec[iRec] + nz * d_x_rec[iRec]] +=
      d_rxz[iRec] * d_res[(iRec) * (nSteps) + (it)];
}

__global__ void source_grad(float *d_szz_adj, float *d_sxx_adj, int nz,
                            float *d_StfGrad, int it, float dt, int z_src,
                            int x_src, double rxz) {
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id == 0) {
    d_StfGrad[it] =
        -(d_szz_adj[z_src + nz * x_src] + rxz * d_sxx_adj[z_src + nz * x_src]) *
        dt;
  } else {
    return;
  }
}

// Dongzhuo Li 01/28/2019
__global__ void cuda_bp_filter1d(int nSteps, float dt, int nrec,
                                 cufftComplex *d_data_F, float f0, float f1,
                                 float f2, float f3) {
  int nf = nSteps / 2 + 1;
  float df = 1.0 / dt / nSteps;

  int idf = blockIdx.x * blockDim.x + threadIdx.x;
  int idr = blockIdx.y * blockDim.y + threadIdx.y;
  int ip = idr * nf + idf;

  float freq = idf * df;

  float filter_amp = 1.0;

  // printf("fffffff = %f\n", freq);

  if (idf >= 0 && idf < nf && idr >= 0 && idr < nrec) {
    if (freq >= f0 && freq < f1) {
      filter_amp = sin(PI / 2.0 * (freq - f0) / (f1 - f0));
    } else if (freq >= f1 && freq < f2) {
      filter_amp = 1.0;
    } else if (freq >= f2 && freq < f3) {
      filter_amp = cos(PI / 2.0 * (freq - f2) / (f3 - f2));
    } else {
      filter_amp = 0.0;
    }

    d_data_F[ip].x *= filter_amp * filter_amp;
    d_data_F[ip].y *= filter_amp * filter_amp;
  }
}

__global__ void cuda_filter1d(int nf, int nrec, cuFloatComplex *d_data_F,
                              cuFloatComplex *d_coef) {
  int idf = blockIdx.x * blockDim.x + threadIdx.x;
  int idr = blockIdx.y * blockDim.y + threadIdx.y;
  int ip = idr * nf + idf;
  if (idf >= 0 && idf < nf && idr >= 0 && idr < nrec) {
    d_data_F[ip] = cuCmulf(d_data_F[ip], d_coef[idf]);
  }
}

__global__ void cuda_normalize(int nz, int nx, float *data, float factor) {
  int idz = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (factor == 0.0) {
    printf("Dividing by zero!\n");
    return;
  }
  if (idz >= 0 && idz < nz && idx >= 0 && idx < nx) {
    data[idx * nz + idz] *= factor;
  } else {
    return;
  }
}

// windowing in the time axis
__global__ void cuda_window(int nt, int nrec, float dt, float *d_win_start,
                            float *d_win_end, float *d_weights,
                            float src_weight, float ratio, float *data) {
  int idt = blockIdx.x * blockDim.x + threadIdx.x;
  int idr = blockIdx.y * blockDim.y + threadIdx.y;
  int ip = idr * nt + idt;

  // stupid bug... (I put the if just befor line 614)
  if (idt >= 0 && idt < nt && idr >= 0 && idr < nrec) {
    float window_amp = 1.0;

    float t = idt * dt;

    if (ratio > 0.5) {
      printf("Dividing by zero!\n");
      return;
    }

    float t0 = d_win_start[idr];
    float t3 = d_win_end[idr];
    if (t0 == 0.0 && t3 == 0.0) printf("t0 = %f, t3 = %f\n\n", t0, t3);

    float t_max = nt * dt;
    if (t0 < 0.0) t0 = 0.0;
    if (t0 > t_max) t0 = t_max;
    if (t3 < 0.0) t3 = 0.0;
    if (t3 > t_max) t3 = t_max;

    float offset = (t3 - t0) * ratio;
    if (offset <= 0.0) {
      printf("Window error 1!!\n");
      printf("offset = %f\n", offset);
      return;
    }

    float t1 = t0 + offset;
    float t2 = t3 - offset;

    if (t >= t0 && t < t1) {
      window_amp = sin(PI / 2.0 * (t - t0) / (t1 - t0));
    } else if (t >= t1 && t < t2) {
      window_amp = 1.0;
    } else if (t >= t2 && t < t3) {
      window_amp = cos(PI / 2.0 * (t - t2) / (t3 - t2));
    } else {
      window_amp = 0.0;
    }

    data[ip] *= window_amp * window_amp * d_weights[idr] * src_weight;
  } else {
    return;
  }
}
// overloaded window function: without specifying windows and weights
__global__ void cuda_window(int nt, int nrec, float dt, float ratio,
                            float *data) {
  int idt = blockIdx.x * blockDim.x + threadIdx.x;
  int idr = blockIdx.y * blockDim.y + threadIdx.y;
  int ip = idr * nt + idt;

  if (idt >= 0 && idt < nt && idr >= 0 && idr < nrec) {
    float window_amp = 1.0;

    float t = idt * dt;

    // if (ratio > 0.5) {
    //     printf("Dividing by zero!\n");
    //     return;
    // }

    float t0 = 0;
    float t3 = nt * dt;

    float offset = nt * dt * ratio;
    if (2.0 * offset >= t3 - t0) {
      printf("Window error 2!\n");
      return;
    }

    float t1 = t0 + offset;
    float t2 = t3 - offset;

    if (t >= t0 && t < t1) {
      window_amp = sin(PI / 2.0 * (t - t0) / (t1 - t0));
    } else if (t >= t1 && t < t2) {
      window_amp = 1.0;
    } else if (t >= t2 && t < t3) {
      window_amp = cos(PI / 2.0 * (t - t2) / (t3 - t2));
    } else {
      window_amp = 0.0;
    }

    data[ip] *= window_amp * window_amp;
  }
}

// Array padding
__global__ void cuda_embed_crop(int nz, int nx, float *d_data, int nz_pad,
                                int nx_pad, float *d_data_pad, bool isEmbed) {
  int idz = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = blockIdx.y * blockDim.y + threadIdx.y;
  int ip = idx * nz + idz;
  int ip_pad = idx * nz_pad + idz;
  if (idz >= 0 && idz < nz && idx >= 0 && idx < nx) {
    if (isEmbed) {
      d_data_pad[ip_pad] = d_data[ip];
    } else {
      d_data[ip] = d_data_pad[ip_pad];
    }
  } else {
    return;
  }
}

// Dongzhuo Li 02/02/2019
__global__ void cuda_spectrum_update(int nf, int nrec,
                                     cuFloatComplex *d_data_obs_F,
                                     cuFloatComplex *d_data_cal_F,
                                     cuFloatComplex *d_source_F,
                                     cuFloatComplex *d_coef) {
  int idr = 0, idf = 0, ip = 0;

  const int Block_Size = 512;
  const float lambda = 1e-6;
  cuFloatComplex c_obs = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex c_cal = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex c_nominator = make_cuFloatComplex(0.0f, 0.0f);
  cuFloatComplex c_denominator = make_cuFloatComplex(0.0f, 0.0f);

  __shared__ cuFloatComplex sh_nominator_F[Block_Size];
  __shared__ cuFloatComplex sh_denominator_F[Block_Size];

  int tid =
      threadIdx.x;  // one thread handles s receivers (with 512 as the interval)
  int bid = blockIdx.x;  // one block handles one frequency
  sh_nominator_F[tid] = make_cuFloatComplex(0.0f, 0.0f);
  sh_denominator_F[tid] = make_cuFloatComplex(0.0f, 0.0f);
  __syncthreads();

  for (int s = 0; s < (nrec + Block_Size - 1) / Block_Size; s++) {
    idr = s * blockDim.x + tid;
    idf = bid;
    ip = idr * nf + idf;
    if (idr >= 0 && idr < nrec && idf >= 0 && idf < nf) {
      c_obs = d_data_obs_F[ip];
      c_cal = d_data_cal_F[ip];
      sh_nominator_F[tid] =
          cuCaddf(sh_nominator_F[tid], cuCmulf(cuConjf(c_cal), c_obs));
      sh_denominator_F[tid] =
          cuCaddf(sh_denominator_F[tid], cuCmulf(cuConjf(c_cal), c_cal));
    }
  }
  __syncthreads();

  // do reduction in shared memory
  for (int s = blockDim.x / 2; s >= 1; s /= 2) {
    if (tid < s) {
      sh_nominator_F[tid] =
          cuCaddf(sh_nominator_F[tid], sh_nominator_F[tid + s]);
      sh_denominator_F[tid] =
          cuCaddf(sh_denominator_F[tid], sh_denominator_F[tid + s]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    sh_denominator_F[0].x += lambda;
    // printf("nomi = %f, deno = %f\n", cuCabsf(sh_nominator_F[0]),
    // cuCabsf(sh_denominator_F[0]));
    sh_nominator_F[0] = cuCdivf(sh_nominator_F[0], sh_denominator_F[0]);
    // printf("coef = %f", sh_nominator_F[0].x);
    d_coef[bid] = sh_nominator_F[0];
    d_source_F[bid] = cuCmulf(d_source_F[bid], sh_nominator_F[0]);
  }
  // printf("tid = %d\n", tid);
  __syncthreads();

  for (int s = 0; s < (nrec + Block_Size - 1) / Block_Size; s++) {
    idr = s * blockDim.x + tid;
    idf = bid;
    ip = idr * nf + idf;
    if (idr >= 0 && idr < nrec && idf >= 0 && idf < nf) {
      d_data_cal_F[ip] = cuCmulf(d_data_cal_F[ip], sh_nominator_F[0]);
      // d_data_cal_F[ip].x *= cuCabsf(sh_nominator_F[0]);
      // d_data_cal_F[ip].y *= cuCabsf(sh_nominator_F[0]);
      // if (tid == 0) printf("ratio = %f\n", cuCabsf(sh_nominator_F[0]));
    }
  }
  __syncthreads();
}

__global__ void cuda_find_absmax(int n, float *data, float *maxval) {
  int tid =
      threadIdx.x;  // one thread handles s receivers (with 512 as the interval)
  const int Block_Size = 512;
  __shared__ float sh_data[Block_Size];
  sh_data[tid] = 0.0;
  __syncthreads();
  for (int s = 0; s < (n + Block_Size - 1) / Block_Size; s++) {
    int ip = s * blockDim.x + tid;
    if (ip >= 0 && ip < n) {
      if (fabs(data[ip]) > fabs(sh_data[tid])) sh_data[tid] = fabs(data[ip]);
    }
  }
  __syncthreads();

  // do reduction in shared memory
  for (int s = blockDim.x / 2; s >= 1; s /= 2) {
    if (tid < s) {
      sh_data[tid] =
          (sh_data[tid] >= sh_data[tid + s]) ? sh_data[tid] : sh_data[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) maxval[0] = sh_data[0];
  __syncthreads();
}

// Dongzhuo Li's last attempt - 09/26/2019
// find out the norm square of each trace for normalization
// number of blocks = number of traces
// size of a block  = 512
__global__ void cuda_find_normfact(int nt, int nrec, float *data1, float *data2,
                                   float *normfact) {
  // one thread handles s time samples (with 512 as the interval)
  int tid = threadIdx.x;
  int bid = blockIdx.x;  // one block handles one trace
  const int Block_Size = 512;
  __shared__ float sh_data[Block_Size];
  sh_data[tid] = 0.0;
  __syncthreads();
  for (int s = 0; s < (nt + Block_Size - 1) / Block_Size; s++) {
    int ip = s * blockDim.x + tid + bid * nt;
    int time_id = s * blockDim.x + tid;
    if (time_id >= 0 && time_id < nt && bid >= 0 && bid < nrec) {
      sh_data[tid] += data1[ip] * data2[ip];
    }
    __syncthreads();
  }

  // do reduction in shared memory
  for (int s = blockDim.x / 2; s >= 1; s /= 2) {
    if (tid < s) {
      sh_data[tid] += sh_data[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    normfact[bid] = sh_data[0] + DIVCONST;  // add a very small number
    // printf("norm = %f\n", sh_data[0]);
  }
  __syncthreads();
}
//
// // normalize each trace by its normfact
// __global__ void cuda_normal_traces(int nt, int nrec, float *normfact,
//                                    float *data) {
//   int idt = blockIdx.x * blockDim.x + threadIdx.x;
//   int idr = blockIdx.y * blockDim.y + threadIdx.y;
//   int ip = idr * nt + idt;

//   if (idt >= 0 && idt < nt && idr >= 0 && idr < nrec) {
//     data[ip] = data[ip] / sqrt(normfact[idr]);
//   } else {
//     return;
//   }
// }
//
// normalized zero-lag cross-correlation misfit function
__global__ void cuda_normal_misfit(int nrec, float *d_cross_normfact,
                                   float *d_obs_normfact, float *d_cal_normfact,
                                   float *misfit, float *d_weights,
                                   float src_weight) {
  // one thread handles s receivers (with 512 as the interval)
  int tid = threadIdx.x;
  const int Block_Size = 512;
  __shared__ float sh_data[Block_Size];
  sh_data[tid] = 0.0;
  __syncthreads();
  for (int s = 0; s < (nrec + Block_Size - 1) / Block_Size; s++) {
    int ip = s * blockDim.x + tid;
    if (ip >= 0 && ip < nrec) {
      sh_data[tid] += d_cross_normfact[ip] /
                      (sqrt(d_obs_normfact[ip]) * sqrt(d_cal_normfact[ip])) *
                      d_weights[ip] * src_weight;
    }
  }
  __syncthreads();

  // do reduction in shared memory
  for (int s = blockDim.x / 2; s >= 1; s /= 2) {
    if (tid < s) {
      sh_data[tid] += sh_data[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) *misfit = -2.0 * sh_data[0];  // since I multiply 0.5 later
  __syncthreads();
}
//
// apply the weighting factor to the residual
__global__ void cuda_normal_adjoint_source(
    int nt, int nrec, float *d_obs_normfact, float *d_cal_normfact,
    float *d_cross_normfact, float *d_data_obs, float *d_data, float *d_res,
    float *d_weights, float src_weight) {
  int idt = blockIdx.x * blockDim.x + threadIdx.x;
  int idr = blockIdx.y * blockDim.y + threadIdx.y;
  int ip = idr * nt + idt;

  if (idt >= 0 && idt < nt && idr >= 0 && idr < nrec) {
    d_res[ip] = (d_data_obs[ip] -
                 d_cross_normfact[idr] / d_cal_normfact[idr] * d_data[ip]) /
                (sqrt(d_obs_normfact[idr]) * sqrt(d_cal_normfact[idr])) *
                d_weights[idr] * src_weight;
    // d_res[ip] = (d_data_obs[ip] - d_data[ip]);
    // if (idt == 0) {
    //   printf("cross-cal-ratio = %f\n",
    //          d_cross_normfact[idr] / d_cal_normfact[idr]);
    // }
  } else {
    return;
  }
}

// 1D band-pass filtering wrapper code
// Steps: padding, FFT, filtering, IFFT, cropping
void bp_filter1d(int nSteps, float dt, int nrec, float *d_data, float *filter) {
  int nSteps_pad = 2 * nSteps;
  int nfft = nSteps_pad / 2 + 1;
  // float df = 1.0/dt/nSteps_pad;
  float *d_data_pad;
  float f0 = filter[0];
  float f1 = filter[1];
  float f2 = filter[2];
  float f3 = filter[3];
  cufftHandle plan_f, plan_b;
  cufftComplex *d_data_F;

  dim3 threads(TX, TY);
  dim3 blocks((nSteps + TX - 1) / TX, (nrec + TY - 1) / TY);

  // float *h_test = new float[nSteps_pad];

  // pad data
  CHECK(cudaMalloc((void **)&d_data_pad, nSteps_pad * nrec * sizeof(float)));
  intialArrayGPU<<<blocks, threads>>>(d_data_pad, nSteps_pad, nrec, 0.0);
  cuda_embed_crop<<<blocks, threads>>>(nSteps, nrec, d_data, nSteps_pad, nrec,
                                       d_data_pad, true);

  // CHECK(cudaMemcpy(h_test, d_data_pad, nSteps * sizeof(float),
  // cudaMemcpyDeviceToHost)); displayArray("h_test", h_test, nSteps_pad, 1);

  // filtering
  CHECK(cudaMalloc((void **)&d_data_F, sizeof(cufftComplex) * nfft * nrec));
  cufftPlan1d(&plan_f, nSteps_pad, CUFFT_R2C, nrec);
  cufftExecR2C(plan_f, d_data_pad, d_data_F);  // forward FFT
  cufftDestroy(plan_f);

  cuda_bp_filter1d<<<blocks, threads>>>(nSteps_pad, dt, nrec, d_data_F, f0, f1,
                                        f2, f3);
  cufftPlan1d(&plan_b, nSteps_pad, CUFFT_C2R, nrec);
  cufftExecC2R(plan_b, d_data_F, d_data_pad);  // inverse FFT
  cufftDestroy(plan_b);

  // CHECK(cudaMemcpy(h_test, d_data_pad, nSteps * sizeof(float),
  // cudaMemcpyDeviceToHost)); displayArray("h_test", h_test, nSteps_pad, 1);

  // crop data
  cuda_embed_crop<<<blocks, threads>>>(nSteps, nrec, d_data, nSteps_pad, nrec,
                                       d_data_pad, false);

  // normalization (in the padded fft, the length is nSteps_pad)
  cuda_normalize<<<blocks, threads>>>(nSteps, nrec, d_data,
                                      1 / float(nSteps_pad));

  CHECK(cudaFree(d_data_F));
  CHECK(cudaFree(d_data_pad));
}

// source signature and calculated data update
// Steps: padding, FFT, compute spectrum, filtering, IFFT, cropping
float source_update(int nSteps, float dt, int nrec, float *d_data_obs,
                    float *d_data_cal, float *d_source,
                    cuFloatComplex *d_coef) {
  int nSteps_pad = 2 * nSteps;
  int nfft = nSteps_pad / 2 + 1;
  float *d_data_obs_pad, *d_data_cal_pad, *d_source_pad;

  cufftHandle plan_f, plan_b;
  cufftComplex *d_data_obs_F, *d_data_cal_F, *d_source_F;

  dim3 threads(TX, TY);
  dim3 blocks((nSteps + TX - 1) / TX, (nrec + TY - 1) / TY);
  dim3 blocks_pad((nSteps_pad + TX - 1) / TX, (nrec + TY - 1) / TY);

  // pad data and window data
  CHECK(
      cudaMalloc((void **)&d_data_obs_pad, nSteps_pad * nrec * sizeof(float)));
  CHECK(
      cudaMalloc((void **)&d_data_cal_pad, nSteps_pad * nrec * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_source_pad, nSteps_pad * sizeof(float)));
  intialArrayGPU<<<blocks, threads>>>(d_data_obs_pad, nSteps_pad, nrec, 0.0);
  intialArrayGPU<<<blocks, threads>>>(d_data_cal_pad, nSteps_pad, nrec, 0.0);
  intialArrayGPU<<<(nSteps_pad + 31) / 32, 32>>>(d_source_pad, nSteps_pad, 1,
                                                 0.0);
  cuda_embed_crop<<<blocks, threads>>>(nSteps, nrec, d_data_obs, nSteps_pad,
                                       nrec, d_data_obs_pad, true);
  cuda_embed_crop<<<blocks, threads>>>(nSteps, nrec, d_data_cal, nSteps_pad,
                                       nrec, d_data_cal_pad, true);
  cuda_window<<<blocks_pad, threads>>>(nSteps_pad, nrec, dt, 0.01,
                                       d_data_obs_pad);
  cuda_window<<<blocks_pad, threads>>>(nSteps_pad, nrec, dt, 0.01,
                                       d_data_cal_pad);

  cuda_embed_crop<<<(nSteps_pad + 31) / 32, 32>>>(
      nSteps, 1, d_source, nSteps_pad, 1, d_source_pad, true);

  // CHECK(cudaMemcpy(h_test, d_data_pad, nSteps * sizeof(float),
  // cudaMemcpyDeviceToHost)); displayArray("h_test", h_test, nSteps_pad, 1);

  // // filtering
  CHECK(cudaMalloc((void **)&d_data_obs_F, sizeof(cufftComplex) * nfft * nrec));
  CHECK(cudaMalloc((void **)&d_data_cal_F, sizeof(cufftComplex) * nfft * nrec));
  CHECK(cudaMalloc((void **)&d_source_F, sizeof(cufftComplex) * nfft))
  cufftPlan1d(&plan_f, nSteps_pad, CUFFT_R2C, nrec);
  cufftExecR2C(plan_f, d_data_obs_pad,
               d_data_obs_F);  // forward FFT of observed data
  cufftExecR2C(plan_f, d_data_cal_pad,
               d_data_cal_F);  // forward FFT of calculated data
  cufftDestroy(plan_f);

  cufftPlan1d(&plan_f, nSteps_pad, CUFFT_R2C, 1);  // source FFT
  cufftExecR2C(plan_f, d_source_pad, d_source_F);
  cufftDestroy(plan_f);

  // cuda_bp_filter1d<<<blocks,threads>>>(nSteps_pad, dt, nrec, d_data_F, f0,
  // f1, f2, f3);
  cuda_spectrum_update<<<nfft, 512>>>(nfft, nrec, d_data_obs_F, d_data_cal_F,
                                      d_source_F, d_coef);

  cufftPlan1d(&plan_b, nSteps_pad, CUFFT_C2R, nrec);
  cufftExecC2R(plan_b, d_data_cal_F, d_data_cal_pad);  // inverse FFT
  cufftDestroy(plan_b);

  cufftPlan1d(&plan_b, nSteps_pad, CUFFT_C2R, 1);
  cufftExecC2R(plan_b, d_source_F, d_source_pad);  // inverse FFT
  cufftDestroy(plan_b);

  // CHECK(cudaMemcpy(h_test, d_data_pad, nSteps * sizeof(float),
  // cudaMemcpyDeviceToHost)); displayArray("h_test", h_test, nSteps_pad, 1);

  // crop data
  cuda_embed_crop<<<blocks, threads>>>(nSteps, nrec, d_data_cal, nSteps_pad,
                                       nrec, d_data_cal_pad, false);
  cuda_embed_crop<<<(nSteps + 31) / 32, 32>>>(nSteps, 1, d_source, nSteps_pad,
                                              1, d_source_pad, false);

  // normalization (in the padded fft, the length is nSteps_pad)
  // printf("amp = %f\n", amp_ratio);
  cuda_normalize<<<blocks, threads>>>(nSteps, nrec, d_data_cal,
                                      1.0f / float(nSteps_pad));
  cuda_normalize<<<(nSteps + 31) / 32, 32>>>(nSteps, 1, d_source,
                                             1.0f / float(nSteps_pad));

  float amp_ratio = amp_ratio_comp(nSteps * nrec, d_data_obs, d_data_cal);
  // cuda_normalize<<<blocks,threads>>>(nSteps, nrec, d_data_cal, amp_ratio);
  // cuda_normalize<<<(nSteps+31)/32, 32>>>(nSteps, 1, d_source,
  // amp_ratio/float(nSteps_pad));

  // // update amplitude
  // cuda_find_absmax<<<1, 512>>>(nSteps*nrec, d_data_obs, d_obs_maxval);
  // cuda_find_absmax<<<1, 512>>>(nSteps*nrec, d_data_cal, d_cal_maxval);
  // CHECK(cudaMemcpy(obs_maxval, d_obs_maxval, sizeof(float),
  // cudaMemcpyDeviceToHost)); CHECK(cudaMemcpy(cal_maxval, d_cal_maxval,
  // sizeof(float), cudaMemcpyDeviceToHost));
  // cuda_normalize<<<blocks,threads>>>(nSteps, nrec,
  // d_data_cal, 1.0/amp_ratio); printf("Shot gather amplitude ratio = %f\n",
  // obs_maxval[0]/cal_maxval[0]);

  CHECK(cudaFree(d_data_obs_pad));
  CHECK(cudaFree(d_data_cal_pad));
  CHECK(cudaFree(d_data_obs_F));
  CHECK(cudaFree(d_data_cal_F));
  CHECK(cudaFree(d_source_pad));
  CHECK(cudaFree(d_source_F));

  return amp_ratio;
}

// source signature and calculated data update
// Steps: padding, FFT, compute spectrum, filtering, IFFT, cropping
void source_update_adj(int nSteps, float dt, int nrec, float *d_data,
                       float amp_ratio, cuFloatComplex *d_coef) {
  int nSteps_pad = 2 * nSteps;
  int nfft = nSteps_pad / 2 + 1;
  float *d_data_pad;

  cufftHandle plan_f, plan_b;
  cufftComplex *d_data_F;

  dim3 threads(TX, TY);
  dim3 blocks((nSteps + TX - 1) / TX, (nrec + TY - 1) / TY);
  dim3 blocks_pad((nSteps_pad + TX - 1) / TX, (nrec + TY - 1) / TY);

  // cuda_normalize<<<blocks,threads>>>(nSteps, nrec, d_data, amp_ratio);

  // pad data
  CHECK(cudaMalloc((void **)&d_data_pad, nSteps_pad * nrec * sizeof(float)));
  intialArrayGPU<<<blocks, threads>>>(d_data_pad, nSteps_pad, nrec, 0.0);
  cuda_embed_crop<<<blocks, threads>>>(nSteps, nrec, d_data, nSteps_pad, nrec,
                                       d_data_pad, true);
  cuda_window<<<blocks_pad, threads>>>(nSteps_pad, nrec, dt, 0.01, d_data_pad);

  CHECK(cudaMalloc((void **)&d_data_F, sizeof(cufftComplex) * nfft * nrec));
  cufftPlan1d(&plan_f, nSteps_pad, CUFFT_R2C, nrec);
  cufftExecR2C(plan_f, d_data_pad, d_data_F);
  cufftDestroy(plan_f);

  // update data
  cuda_filter1d<<<blocks, threads>>>(nfft, nrec, d_data_F, d_coef);

  cufftPlan1d(&plan_b, nSteps_pad, CUFFT_C2R, nrec);
  cufftExecC2R(plan_b, d_data_F, d_data_pad);  // inverse FFT
  cufftDestroy(plan_b);

  // crop data
  cuda_embed_crop<<<blocks, threads>>>(nSteps, nrec, d_data, nSteps_pad, nrec,
                                       d_data_pad, false);

  // normalization (in the padded fft, the length is nSteps_pad)
  // printf("amp_adj = %f\n", amp_ratio);
  cuda_normalize<<<blocks, threads>>>(nSteps, nrec, d_data,
                                      amp_ratio / float(nSteps_pad));

  CHECK(cudaFree(d_data_pad));
  CHECK(cudaFree(d_data_F));
}

float amp_ratio_comp(int n, float *d_data_obs, float *d_data_cal) {
  float *obs_maxval = nullptr, *cal_maxval = nullptr;
  float *d_obs_maxval, *d_cal_maxval;

  obs_maxval = (float *)malloc(sizeof(float));
  cal_maxval = (float *)malloc(sizeof(float));
  CHECK(cudaMalloc((void **)&d_obs_maxval, sizeof(float)));
  CHECK(cudaMalloc((void **)&d_cal_maxval, sizeof(float)));

  cuda_find_absmax<<<1, 512>>>(n, d_data_obs, d_obs_maxval);
  cuda_find_absmax<<<1, 512>>>(n, d_data_cal, d_cal_maxval);
  CHECK(cudaMemcpy(obs_maxval, d_obs_maxval, sizeof(float),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(cal_maxval, d_cal_maxval, sizeof(float),
                   cudaMemcpyDeviceToHost));
  // printf("Shot gather amplitude ratio = %f\n",
  // obs_maxval[0]/cal_maxval[0]);

  float ratio = 0.0;
  if (cal_maxval[0] != 0.0) {
    ratio = obs_maxval[0] / cal_maxval[0];
  }

  CHECK(cudaFree(d_obs_maxval));
  CHECK(cudaFree(d_cal_maxval));
  delete[] obs_maxval;
  delete[] cal_maxval;

  return ratio;
}