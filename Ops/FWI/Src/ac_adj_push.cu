#define d_vx(z,x)  d_vx[(x)*(nz)+(z)]
#define d_vy(z,x)  d_vy[(x)*(nz)+(z)]
#define d_vz(z,x)  d_vz[(x)*(nz)+(z)]
#define d_szz(z,x) d_szz[(x)*(nz)+(z)] // Pressure
#define d_mem_dszz_dz(z,x) d_mem_dszz_dz[(x)*(nz)+(z)]
#define d_mem_dsxx_dx(z,x) d_mem_dsxx_dx[(x)*(nz)+(z)]
#define d_mem_dvz_dz(z,x) d_mem_dvz_dz[(x)*(nz)+(z)]
#define d_mem_dvx_dx(z,x) d_mem_dvx_dx[(x)*(nz)+(z)]
#define d_Lambda(z,x)     d_Lambda[(x)*(nz)+(z)]
#define d_Den(z,x)        d_Den[(x)*(nz)+(z)]
#define d_ave_Byc_a(z,x) d_ave_Byc_a[(x)*(nz)+(z)]
#define d_ave_Byc_b(z,x) d_ave_Byc_b[(x)*(nz)+(z)]
#define d_adj_temp(z,x)  d_adj_temp[(x)*(nz)+(z)]
#include "utilities.h"

__global__ void ac_adj_push(float *d_vz, float *d_vx, float *d_szz, float *d_adj_temp,\
	float *d_mem_dvz_dz, float *d_mem_dvx_dx, float *d_mem_dszz_dz, float *d_mem_dsxx_dx, \
	float *d_Lambda, float *d_Den, float *d_ave_Byc_a, float *d_ave_Byc_b,\
	float *d_K_z_half, float *d_a_z_half, float *d_b_z_half, \
	float *d_K_x_half, float *d_a_x_half, float *d_b_x_half, \
	float *d_K_z, float *d_a_z, float *d_b_z, \
	float *d_K_x, float *d_a_x, float *d_b_x, \
	int nz, int nx, float dt, float dz, float dx, int nPml, int nPad){


	int hFD = 2;
	int n_tile_z = blockDim.x - 2*hFD;
	int n_tile_x = blockDim.y - 2*hFD;

	int tz = threadIdx.x; // local thread coordinates
	int tx = threadIdx.y;
	int localz = blockDim.x;
  int gidz = blockIdx.x*n_tile_z + threadIdx.x - hFD;
  int gidx = blockIdx.y*n_tile_x + threadIdx.y - hFD;
  float c1 = 9.0/8.0;
  float c2 = 1.0/24.0;

  __shared__ float sh_d[720];


	sh_d[tz + tx*localz] = 0.0;

	__syncthreads();


if(gidz>=0 & gidz<=nz-nPad-1 && gidx>=0 && gidx<=nx-1) {

// // ======================== vz
		d_adj_temp(gidz, gidx) = 0.0;
		__syncthreads();
		if(tz >=hFD && tz < n_tile_z+hFD && tx >=hFD && tx < n_tile_x+hFD){
			if(gidz>=2 & gidz<=nz-nPad-3 && gidx>=2 && gidx<=nx-3) {
				// update vz
				// printf("tz = %d   \n", tz);
				float dvzC1 = d_Lambda(gidz, gidx)/d_K_z_half[gidz]*dt/dz*d_szz(gidz, gidx);
				float dvzC2 = d_a_z_half[gidz]*d_mem_dvz_dz(gidz, gidx);
				atomicAdd(&sh_d[tz +  tx*localz], - c1*dvzC1 - c1*dvzC2);
				atomicAdd(&sh_d[tz+1 +  tx*localz], +c1*dvzC1 + c1*dvzC2);
				atomicAdd(&sh_d[tz+2 +  tx*localz], -c2*dvzC1 - c2*dvzC2);
				atomicAdd(&sh_d[tz-1 +  tx*localz], c2*dvzC1 + c2*dvzC2);
			}
		}
		__syncthreads();

		atomicAdd(&d_vz(gidz, gidx), sh_d[tz + tx*localz]);
		__syncthreads();

		sh_d[tz + tx*localz] = 0.0;
	  __syncthreads();

// ======================== vx
		__syncthreads();
		if(tz >=hFD && tz < n_tile_z+hFD && tx >=hFD && tx < n_tile_x+hFD){
			if(gidz>=2 & gidz<=nz-nPad-3 && gidx>=2 && gidx<=nx-3) {
				// update vx
				float dvxC1 = d_Lambda(gidz, gidx)/d_K_x[gidx]*dt/dx*d_szz(gidz, gidx);
				float dvxC2 = d_a_x[gidx]*d_mem_dvx_dx(gidz, gidx);
	
				atomicAdd(&sh_d[tz + tx*localz], (c1*dvxC1 + c1*dvxC2));
				atomicAdd(&sh_d[tz + (tx-1)*localz], (-c1*dvxC1 - c1*dvxC2));
				atomicAdd(&sh_d[tz + (tx+1)*localz], (-c2*dvxC1 - c2*dvxC2));
				atomicAdd(&sh_d[tz + (tx-2)*localz], (+c2*dvxC1 + c2*dvxC2));
			}
		}
		__syncthreads();

		atomicAdd(&d_vx(gidz, gidx), sh_d[tz + tx*localz]);

		sh_d[tz + tx*localz] = 0.0;
	  __syncthreads();



//  ======================== d_mem_dsxx_dx...
		if(gidz>=2 & gidz<=nz-nPad-3 && gidx>=2 && gidx<=nx-3) {
			// update d_mem_dsxx_dx...
			d_mem_dszz_dz(gidz, gidx) = d_b_z[gidz]*d_mem_dszz_dz(gidz, gidx) + d_ave_Byc_a(gidz, gidx)*d_vz(gidz, gidx)*dt/dz;
			d_mem_dsxx_dx(gidz, gidx) = d_b_x_half[gidx]*d_mem_dsxx_dx(gidz, gidx) + d_ave_Byc_b(gidz, gidx)*d_vx(gidz, gidx)*dt/dx;
		}


// ======================== stress
		__syncthreads();
		if(tz >=hFD && tz < n_tile_z+hFD && tx >=hFD && tx < n_tile_x+hFD){
			if(gidz>=2 & gidz<=nz-nPad-3 && gidx>=2 && gidx<=nx-3) {
				// update stress
				float dzzC1 = d_ave_Byc_a(gidz, gidx)/d_K_z[gidz]*dt/dz*d_vz(gidz, gidx);
				float dzzC3 = d_a_z[gidz]*d_mem_dszz_dz(gidz, gidx);
				float dzzC2 = d_ave_Byc_b(gidz, gidx)/d_K_x_half[gidx]*dt/dx*d_vx(gidz, gidx);
				float dzzC4 = d_a_x_half[gidx]*d_mem_dsxx_dx(gidz, gidx);
				atomicAdd(&sh_d[tz + tx*localz], c1*dzzC1 - c1*dzzC2 + c1*dzzC3 - c1*dzzC4);
				atomicAdd(&sh_d[tz-1 + (tx)*localz], (-c1*dzzC1 - c1*dzzC3));
				atomicAdd(&sh_d[tz-2 + (tx)*localz], (c2*dzzC1 + c2*dzzC3));
				atomicAdd(&sh_d[tz+1 + (tx)*localz], (-c2*dzzC1 - c2*dzzC3));
				atomicAdd(&sh_d[tz + (tx-1)*localz], (c2*dzzC2 + c2*dzzC4));
				atomicAdd(&sh_d[tz + (tx+1)*localz], (c1*dzzC2 + c1*dzzC4));
				atomicAdd(&sh_d[tz + (tx+2)*localz], (-c2*dzzC2 - c2*dzzC4));
			}
		}

			__syncthreads();
		atomicAdd(&d_szz(gidz, gidx), sh_d[tz + tx*localz]);
		
		sh_d[tz + tx*localz] = 0.0;
	  __syncthreads();


//  ======================== d_mem_dvz_dz
		if(gidz>=2 & gidz<=nz-nPad-3 && gidx>=2 && gidx<=nx-3) {
		  // update psi
			d_mem_dvz_dz(gidz, gidx) = d_b_z_half[gidz]*d_mem_dvz_dz(gidz, gidx) + d_Lambda(gidz, gidx)*d_szz(gidz, gidx)*dt/dz;
			d_mem_dvx_dx(gidz, gidx) = d_b_x[gidx]*d_mem_dvx_dx(gidz, gidx) + d_Lambda(gidz, gidx)*d_szz(gidz, gidx)*dt/dx;
		}

	}

	else {
		return;
	}


}
