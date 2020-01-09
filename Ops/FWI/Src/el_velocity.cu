#define d_vx(z,x)  d_vx[(x)*(nz)+(z)]
#define d_vy(z,x)  d_vy[(x)*(nz)+(z)]
#define d_vz(z,x)  d_vz[(x)*(nz)+(z)]
#define d_sxx(z,x) d_sxx[(x)*(nz)+(z)]
#define d_szz(z,x) d_szz[(x)*(nz)+(z)]
#define d_sxz(z,x) d_sxz[(x)*(nz)+(z)]
#define d_vz_adj(z,x)  d_vz_adj[(x)*(nz)+(z)]
#define d_vx_adj(z,x)  d_vx_adj[(x)*(nz)+(z)]
#define d_mem_dszz_dz(z,x) d_mem_dszz_dz[(x)*(nz)+(z)]
#define d_mem_dsxz_dx(z,x) d_mem_dsxz_dx[(x)*(nz)+(z)]
#define d_mem_dsxz_dz(z,x) d_mem_dsxz_dz[(x)*(nz)+(z)]
#define d_mem_dsxx_dx(z,x) d_mem_dsxx_dx[(x)*(nz)+(z)]
#define d_Lambda(z,x)     d_Lambda[(x)*(nz)+(z)]
#define d_Mu(z,x)         d_Mu[(x)*(nz)+(z)]
#define d_Den(z,x)        d_Den[(x)*(nz)+(z)]
#define d_ave_Byc_a(z,x) d_ave_Byc_a[(x)*(nz)+(z)]
#define d_ave_Byc_b(z,x) d_ave_Byc_b[(x)*(nz)+(z)]
#define d_DenGrad(z,x)  d_DenGrad[(x)*(nz)+(z)]
#include<stdio.h>

__global__ void el_velocity(float *d_vz, float *d_vx, float *d_szz, \
	float *d_sxx, float *d_sxz, float *d_mem_dszz_dz, float *d_mem_dsxz_dx, \
	float *d_mem_dsxz_dz, float *d_mem_dsxx_dx, float *d_Lambda, float *d_Mu, \
	float *d_ave_Byc_a, float *d_ave_Byc_b, float *d_K_z, float *d_a_z, float *d_b_z, \
	float *d_K_z_half, 	float *d_a_z_half, float *d_b_z_half, float *d_K_x, float *d_a_x, \
	float *d_b_x, float *d_K_x_half, float *d_a_x_half, float *d_b_x_half, \
	int nz, int nx, float dt, float dz, float dx, int nPml, int nPad, bool isFor, \
	float *d_vz_adj, float *d_vx_adj, float *d_DenGrad){


	int gidz = blockIdx.x*blockDim.x + threadIdx.x;
  int gidx = blockIdx.y*blockDim.y + threadIdx.y;


  float dszz_dz = 0.0;
  float dsxz_dx = 0.0;
  float dsxz_dz = 0.0;
  float dsxx_dx = 0.0;

  float c1 = 9.0/8.0;
  float c2 = 1.0/24.0;
  // float c1 = coef[0];
  // float c2 = coef[1];

  if (isFor) {

	  if(gidz>=2 && gidz<=nz-nPad-3 && gidx>=2 && gidx<=nx-3) {
		  // update vz
			dszz_dz = (c1*(d_szz(gidz+1,gidx)-d_szz(gidz,gidx)) - c2*(d_szz(gidz+2,gidx)-d_szz(gidz-1,gidx)))/dz;
			dsxz_dx = (c1*(d_sxz(gidz,gidx)-d_sxz(gidz,gidx-1)) - c2*(d_sxz(gidz,gidx+1)-d_sxz(gidz,gidx-2)))/dx;

			if(gidz<nPml || (gidz>nz-nPml-nPad-1)){
				d_mem_dszz_dz(gidz,gidx) = d_b_z_half[gidz]*d_mem_dszz_dz(gidz,gidx) + d_a_z_half[gidz]*dszz_dz;
				dszz_dz = dszz_dz / d_K_z_half[gidz] + d_mem_dszz_dz(gidz,gidx);
		  }
		  if(gidx<nPml || gidx>nx-nPml){
				d_mem_dsxz_dx(gidz,gidx) = d_b_x[gidx]*d_mem_dsxz_dx(gidz,gidx) + d_a_x[gidx]*dsxz_dx;
				dsxz_dx = dsxz_dx / d_K_x[gidx] + d_mem_dsxz_dx(gidz,gidx);
			}

			d_vz(gidz,gidx) += (dszz_dz + dsxz_dx) * d_ave_Byc_a(gidz, gidx) * dt;

			// update vx
			dsxz_dz = (c1*(d_sxz(gidz,gidx)-d_sxz(gidz-1,gidx)) - c2*(d_sxz(gidz+1,gidx)-d_sxz(gidz-2,gidx)))/dz;
			dsxx_dx = (c1*(d_sxx(gidz,gidx+1)-d_sxx(gidz,gidx)) - c2*(d_sxx(gidz,gidx+2)-d_sxx(gidz,gidx-1)))/dx;

			if(gidz<nPml || (gidz>nz-nPml-nPad-1)){
				d_mem_dsxz_dz(gidz,gidx) = d_b_z[gidz]*d_mem_dsxz_dz(gidz,gidx) + d_a_z[gidz]*dsxz_dz;
				dsxz_dz = dsxz_dz / d_K_z[gidz] + d_mem_dsxz_dz(gidz,gidx);
			}
			if(gidx<nPml || gidx>nx-nPml){
				d_mem_dsxx_dx(gidz,gidx) = d_b_x_half[gidx]*d_mem_dsxx_dx(gidz,gidx) + d_a_x_half[gidx]*dsxx_dx;	
				dsxx_dx = dsxx_dx / d_K_x_half[gidx] + d_mem_dsxx_dx(gidz,gidx);
			}

			d_vx(gidz,gidx) += (dsxz_dz + dsxx_dx) * d_ave_Byc_b(gidz, gidx) * dt;

		}
		else{
			return;
		}
	} 

	else {

	// ========================================BACKWARD PROPAGATION====================================
	  if(gidz>=nPml && gidz<=nz-nPad-1-nPml && gidx>=nPml && gidx<=nx-1-nPml) {
		  // update vz
			dszz_dz = (c1*(d_szz(gidz+1,gidx)-d_szz(gidz,gidx)) - c2*(d_szz(gidz+2,gidx)-d_szz(gidz-1,gidx)))/dz;
			dsxz_dx = (c1*(d_sxz(gidz,gidx)-d_sxz(gidz,gidx-1)) - c2*(d_sxz(gidz,gidx+1)-d_sxz(gidz,gidx-2)))/dx;

			d_vz(gidz,gidx) -= (dszz_dz + dsxz_dx) * d_ave_Byc_a(gidz, gidx) * dt;

			// update vx
			dsxz_dz = (c1*(d_sxz(gidz,gidx)-d_sxz(gidz-1,gidx)) - c2*(d_sxz(gidz+1,gidx)-d_sxz(gidz-2,gidx)))/dz;
			dsxx_dx = (c1*(d_sxx(gidz,gidx+1)-d_sxx(gidz,gidx)) - c2*(d_sxx(gidz,gidx+2)-d_sxx(gidz,gidx-1)))/dx;

			d_vx(gidz,gidx) -= (dsxz_dz + dsxx_dx) * d_ave_Byc_b(gidz, gidx) * dt;

			// computate the density kernel (spray)
			float grad_ave_Byc_a = -d_vz_adj(gidz,gidx)*(dszz_dz + dsxz_dx)*dt \
				* (-pow(d_ave_Byc_a(gidz,gidx),2)/2.0);
			float grad_ave_Byc_b = -d_vx_adj(gidz,gidx)*(dsxz_dz + dsxx_dx)*dt \
				* (-pow(d_ave_Byc_b(gidz,gidx),2)/2.0);
			atomicAdd(&d_DenGrad[gidz+nz*gidx], grad_ave_Byc_a);
			atomicAdd(&d_DenGrad[gidz+nz*gidx], grad_ave_Byc_b);
			if (gidz+1<=nz-nPad-1-nPml) 
				atomicAdd(&d_DenGrad[gidz+1+nz*gidx], grad_ave_Byc_a);
			if (gidx+1<=gidx<=nx-1-nPml) 
			atomicAdd(&d_DenGrad[gidz+nz*(gidx+1)], grad_ave_Byc_b);


		}
		else{
			return;
		}
	}

}