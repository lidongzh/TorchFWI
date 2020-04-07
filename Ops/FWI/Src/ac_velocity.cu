#define d_vx(z,x)  d_vx[(x)*(nz)+(z)]
#define d_vy(z,x)  d_vy[(x)*(nz)+(z)]
#define d_vz(z,x)  d_vz[(x)*(nz)+(z)]
#define d_szz(z,x) d_szz[(x)*(nz)+(z)] // Pressure
#define d_mem_dszz_dz(z,x) d_mem_dszz_dz[(x)*(nz)+(z)]
#define d_mem_dsxx_dx(z,x) d_mem_dsxx_dx[(x)*(nz)+(z)]
#define d_Lambda(z,x)     d_Lambda[(x)*(nz)+(z)]
#define d_Den(z,x)        d_Den[(x)*(nz)+(z)]
#define d_ave_Byc_a(z,x) d_ave_Byc_a[(x)*(nz)+(z)]
#define d_ave_Byc_b(z,x) d_ave_Byc_b[(x)*(nz)+(z)]
#include<stdio.h>

__global__ void ac_velocity(float *d_vz, float *d_vx, float *d_szz, \
	float *d_mem_dszz_dz, float *d_mem_dsxx_dx, float *d_Lambda, \
	float *d_Den, float *d_ave_Byc_a, float *d_ave_Byc_b, float *d_K_z, \
	float *d_a_z, float *d_b_z, \
	float *d_K_x_half, float *d_a_x_half, float *d_b_x_half, \
	int nz, int nx, float dt, float dz, float dx, int nPml, int nPad, bool isFor){


	int gidz = blockIdx.x*blockDim.x + threadIdx.x;
  int gidx = blockIdx.y*blockDim.y + threadIdx.y;


  float dszz_dz = 0.0;
  float dsxx_dx = 0.0;

  float c1 = 9.0/8.0;
  float c2 = 1.0/24.0;

  if (isFor) {

	  if(gidz>=2 && gidz<=nz-nPad-3 && gidx>=2 && gidx<=nx-3) {

	  	// if(gidz>=2 && gidz<=nz-nPad-2) {
			  // update vz
				dszz_dz = (c1*(d_szz(gidz,gidx)-d_szz(gidz-1,gidx)) - c2*(d_szz(gidz+1,gidx)-d_szz(gidz-2,gidx)))/dz;

				if(gidz<=nPml || (gidz>=nz-nPml-nPad-1)){
					d_mem_dszz_dz(gidz,gidx) = d_b_z[gidz]*d_mem_dszz_dz(gidz,gidx) + d_a_z[gidz]*dszz_dz;
			  }

				d_vz(gidz,gidx) += (dszz_dz/d_K_z[gidz] + d_mem_dszz_dz(gidz,gidx)) * d_ave_Byc_a(gidz, gidx) * dt;
			// }

			// if(gidx>=1 && gidx<=nx-3) {
				// update vx
				dsxx_dx = (c1*(d_szz(gidz,gidx+1)-d_szz(gidz,gidx)) - c2*(d_szz(gidz,gidx+2)-d_szz(gidz,gidx-1)))/dx;

				if(gidx<=nPml || gidx>=nx-nPml-1){
					d_mem_dsxx_dx(gidz,gidx) = d_b_x_half[gidx]*d_mem_dsxx_dx(gidz,gidx) + d_a_x_half[gidx]*dsxx_dx;	
				}

				d_vx(gidz,gidx) += (dsxx_dx/d_K_x_half[gidx] + d_mem_dsxx_dx(gidz,gidx)) * d_ave_Byc_b(gidz, gidx) * dt;
			// }

		}

		else {
			return;
		}

	}

	else {
	  if(gidz>=nPml+2 && gidz<=nz-nPad-3-nPml && gidx>=nPml+2 && gidx<=nx-3-nPml) {

			// update vx
			dsxx_dx = (c1*(d_szz(gidz,gidx+1)-d_szz(gidz,gidx)) - c2*(d_szz(gidz,gidx+2)-d_szz(gidz,gidx-1)))/dx;

			d_vx(gidz,gidx) -= dsxx_dx * d_ave_Byc_b(gidz, gidx) * dt;

		  // update vz
			dszz_dz = (c1*(d_szz(gidz,gidx)-d_szz(gidz-1,gidx)) - c2*(d_szz(gidz+1,gidx)-d_szz(gidz-2,gidx)))/dz;

			d_vz(gidz,gidx) -= dszz_dz * d_ave_Byc_a(gidz, gidx) * dt;

		}

		else {
			return;
		}
	}
}