#define d_vx(z,x)  d_vx[(x)*(nz)+(z)]
#define d_vz(z,x)  d_vz[(x)*(nz)+(z)]
#define d_sxx(z,x) d_sxx[(x)*(nz)+(z)]
#define d_szz(z,x) d_szz[(x)*(nz)+(z)]
#define d_sxz(z,x) d_sxz[(x)*(nz)+(z)]
#define d_mem_dszz_dz(z,x) d_mem_dszz_dz[(x)*(nz)+(z)]
#define d_mem_dsxz_dx(z,x) d_mem_dsxz_dx[(x)*(nz)+(z)]
#define d_mem_dsxz_dz(z,x) d_mem_dsxz_dz[(x)*(nz)+(z)]
#define d_mem_dsxx_dx(z,x) d_mem_dsxx_dx[(x)*(nz)+(z)]
#define d_mem_dvz_dz(z,x) d_mem_dvz_dz[(x)*(nz)+(z)]
#define d_mem_dvz_dx(z,x) d_mem_dvz_dx[(x)*(nz)+(z)]
#define d_mem_dvx_dz(z,x) d_mem_dvx_dz[(x)*(nz)+(z)]
#define d_mem_dvx_dx(z,x) d_mem_dvx_dx[(x)*(nz)+(z)]
#define d_Lambda(z,x)     d_Lambda[(x)*(nz)+(z)]
#define d_Den(z,x)        d_Den[(x)*(nz)+(z)]
#define d_Mu(z,x)         d_Mu[(x)*(nz)+(z)]
#define d_ave_Mu(z,x)     d_ave_Mu[(x)*(nz)+(z)]
#define d_ave_Byc_a(z,x) d_ave_Byc_a[(x)*(nz)+(z)]
#define d_ave_Byc_b(z,x) d_ave_Byc_b[(x)*(nz)+(z)]
#include<stdio.h>

__global__ void el_velocity_adj(
  float *d_vz, float *d_vx, float *d_szz, float *d_sxx, float *d_sxz, \
  float *d_mem_dszz_dz, float *d_mem_dsxz_dx, float *d_mem_dsxz_dz, float *d_mem_dsxx_dx, \
  float *d_mem_dvz_dz, float *d_mem_dvz_dx, float *d_mem_dvx_dz, float *d_mem_dvx_dx, \
  float *d_Lambda, float *d_Mu, float *d_ave_Mu, float *d_Den, float *d_ave_Byc_a, float *d_ave_Byc_b, \
	float *d_K_z_half, float *d_a_z_half, float *d_b_z_half, \
	float *d_K_x_half, float *d_a_x_half, float *d_b_x_half, \
	float *d_K_z, float *d_a_z, float *d_b_z, \
	float *d_K_x, float *d_a_x, float *d_b_x, \
	int nz, int nx, float dt, float dz, float dx, int nPml, int nPad){


	int gidz = blockIdx.x*blockDim.x + threadIdx.x;
  int gidx = blockIdx.y*blockDim.y + threadIdx.y;


  float dpsixx_dx = 0.0;
	float dszz_dx = 0.0;
	float dsxx_dx = 0.0;
	float dpsixz_dz = 0.0;
	float dsxz_dz = 0.0;
	float dpsizz_dz = 0.0;
	float dszz_dz = 0.0;
	float dsxx_dz = 0.0;
	float dpsizx_dx = 0.0;
	float dsxz_dx = 0.0;

  float c1 = 9.0/8.0;
	float c2 = 1.0/24.0;
	
	float lambda = d_Lambda(gidz,gidx);
  float mu = d_Mu(gidz,gidx);



  if(gidz>=2 && gidz<=nz-nPad-3 && gidx>=2 && gidx<=nx-3) {


		// update vx
		dpsixx_dx = (-c1*(d_mem_dvx_dx(gidz,gidx+1)-d_mem_dvx_dx(gidz,gidx)) \
				+ c2*(d_mem_dvx_dx(gidz,gidx+2)-d_mem_dvx_dx(gidz,gidx-1)))/dx;
		dszz_dx = (-c1*(d_szz(gidz,gidx+1)-d_szz(gidz,gidx)) + c2*(d_szz(gidz,gidx+2)-d_szz(gidz,gidx-1)))/dx;
		dsxx_dx = (-c1*(d_sxx(gidz,gidx+1)-d_sxx(gidz,gidx)) + c2*(d_sxx(gidz,gidx+2)-d_sxx(gidz,gidx-1)))/dx;
		dpsixz_dz = (-c1*(d_mem_dvx_dz(gidz,gidx)-d_mem_dvx_dz(gidz-1,gidx)) \
				+ c2*(d_mem_dvx_dz(gidz+1,gidx)-d_mem_dvx_dz(gidz-2,gidx)))/dz;
		dsxz_dz = (-c1*(d_sxz(gidz,gidx)-d_sxz(gidz-1,gidx)) + c2*(d_sxz(gidz+1,gidx)-d_sxz(gidz-2,gidx)))/dz;

		d_vx(gidz, gidx) += (d_a_x[gidx]*dpsixx_dx + lambda*dszz_dx/d_K_x[gidx]*dt \
				+ (lambda+2.0*mu)*dsxx_dx/d_K_x[gidx]*dt + d_a_z_half[gidz]*dpsixz_dz \
				+ d_ave_Mu(gidz,gidx)/d_K_z_half[gidz]*dsxz_dz*dt);

		//update phi_xx_x and phi_xz_z
		if(gidx<nPml || gidx>nx-nPml-1){
			d_mem_dsxx_dx(gidz, gidx) = d_b_x_half[gidx]*d_mem_dsxx_dx(gidz, gidx) + d_ave_Byc_b(gidz, gidx)*d_vx(gidz, gidx)*dt;
		}
		if(gidz<nPml || (gidz>nz-nPml-nPad-1)){
			d_mem_dsxz_dz(gidz, gidx) = d_b_z[gidz]*d_mem_dsxz_dz(gidz, gidx) + d_ave_Byc_b(gidz, gidx)*d_vx(gidz, gidx)*dt;
		}

	  // update vz
		dpsizz_dz = (-c1*(d_mem_dvz_dz(gidz+1,gidx)-d_mem_dvz_dz(gidz,gidx)) \
			+ c2*(d_mem_dvz_dz(gidz+2,gidx)-d_mem_dvz_dz(gidz-1,gidx)))/dz;
		dszz_dz = (-c1*(d_szz(gidz+1,gidx)-d_szz(gidz,gidx)) + c2*(d_szz(gidz+2,gidx)-d_szz(gidz-1,gidx)))/dz;
		dsxx_dz = (-c1*(d_sxx(gidz+1,gidx)-d_sxx(gidz,gidx)) + c2*(d_sxx(gidz+2,gidx)-d_sxx(gidz-1,gidx)))/dz;
		dpsizx_dx = (-c1*(d_mem_dvz_dx(gidz,gidx)-d_mem_dvz_dx(gidz,gidx-1)) \
			+ c2*(d_mem_dvz_dx(gidz,gidx+1)-d_mem_dvz_dx(gidz,gidx-2)))/dx;
		dsxz_dx = (-c1*(d_sxz(gidz,gidx)-d_sxz(gidz,gidx-1)) + c2*(d_sxz(gidz,gidx+1)-d_sxz(gidz,gidx-2)))/dx;

		d_vz(gidz, gidx) += (d_a_z[gidz]*dpsizz_dz + (lambda+2.0*mu)*dszz_dz/d_K_z[gidz]*dt \
			+ lambda*dsxx_dz/d_K_z[gidz]*dt + d_a_x_half[gidx]*dpsizx_dx \
			+ d_ave_Mu(gidz,gidx)/d_K_x_half[gidx]*dsxz_dx*dt);

		// update phi_xz_x and phi_zz_z
		if(gidx<nPml || gidx>nx-nPml-1){
			d_mem_dsxz_dx(gidz, gidx) = d_b_x[gidx]*d_mem_dsxz_dx(gidz, gidx) + d_ave_Byc_a(gidz, gidx)*d_vz(gidz, gidx)*dt;
		}
		if(gidz<nPml || (gidz>nz-nPml-nPad-1)){
			d_mem_dszz_dz(gidz, gidx) = d_b_z_half[gidz]*d_mem_dszz_dz(gidz, gidx) + d_ave_Byc_a(gidz, gidx)*d_vz(gidz, gidx)*dt;
		}

	}

	else {
		return;
	}

}