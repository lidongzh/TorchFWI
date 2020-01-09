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


__global__ void el_stress_adj(
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


	float dphi_xz_x_dx = 0.0;
	float dvz_dx = 0.0;
	float dphi_xz_z_dz = 0.0;
	float dvx_dz = 0.0;
	float dphi_xx_x_dx = 0.0;
	float dvx_dx = 0.0;
	float dphi_zz_z_dz = 0.0;
	float dvz_dz = 0.0;

  float c1 = 9.0/8.0;
	float c2 = 1.0/24.0;
	
	float lambda = d_Lambda(gidz,gidx);
  float mu = d_Mu(gidz,gidx);


	if (gidz>=2 && gidz<=nz-nPad-3 && gidx>=2 && gidx<=nx-3) {

		dphi_xz_x_dx = (-c1*(d_mem_dsxz_dx(gidz,gidx+1)-d_mem_dsxz_dx(gidz,gidx)) \
				+ c2*(d_mem_dsxz_dx(gidz,gidx+2)-d_mem_dsxz_dx(gidz,gidx-1)))/dx;
		dvz_dx = (-c1*(d_vz(gidz,gidx+1)-d_vz(gidz,gidx)) + c2*(d_vz(gidz,gidx+2)-d_vz(gidz,gidx-1)))/dx;
		dphi_xz_z_dz = (-c1*(d_mem_dsxz_dz(gidz+1,gidx)-d_mem_dsxz_dz(gidz,gidx)) \
				+ c2*(d_mem_dsxz_dz(gidz+2,gidx)-d_mem_dsxz_dz(gidz-1,gidx)))/dz;
		dvx_dz = (-c1*(d_vx(gidz+1,gidx)-d_vx(gidz,gidx)) + c2*(d_vx(gidz+2,gidx)-d_vx(gidz-1,gidx)))/dz;

		// update sxz
		d_sxz(gidz,gidx) += d_a_x[gidx]*dphi_xz_x_dx + dvz_dx/d_K_x[gidx]*d_ave_Byc_a(gidz,gidx)*dt \
				+ d_a_z[gidz]*dphi_xz_z_dz + dvx_dz/d_K_z[gidz]*d_ave_Byc_b(gidz,gidx)*dt;

		// update psi_zx and psi_xz
		// if(gidx<nPml || gidx>nx-nPml-1){
			d_mem_dvz_dx(gidz,gidx) = d_b_x_half[gidx]*d_mem_dvz_dx(gidz,gidx) + d_sxz(gidz,gidx)*d_ave_Mu(gidz,gidx)*dt;
		// }
		// if(gidz<nPml || gidz>nz-nPml-nPad-1){
			d_mem_dvx_dz(gidz,gidx) = d_b_z_half[gidz]*d_mem_dvx_dz(gidz,gidx) + d_sxz(gidz,gidx)*d_ave_Mu(gidz,gidx)*dt;
		// }
		  
		dphi_xx_x_dx = (-c1*(d_mem_dsxx_dx(gidz,gidx)-d_mem_dsxx_dx(gidz,gidx-1)) \
				+ c2*(d_mem_dsxx_dx(gidz,gidx+1)-d_mem_dsxx_dx(gidz,gidx-2)))/dx;
		dvx_dx = (-c1*(d_vx(gidz,gidx)-d_vx(gidz,gidx-1)) + c2*(d_vx(gidz,gidx+1)-d_vx(gidz,gidx-2)))/dx;
		dphi_zz_z_dz = (-c1*(d_mem_dszz_dz(gidz,gidx)-d_mem_dszz_dz(gidz-1,gidx)) \
				+ c2*(d_mem_dszz_dz(gidz+1,gidx)-d_mem_dszz_dz(gidz-2,gidx)))/dz;
		dvz_dz = (-c1*(d_vz(gidz,gidx)-d_vz(gidz-1,gidx)) + c2*(d_vz(gidz+1,gidx)-d_vz(gidz-2,gidx)))/dz;

		// update sxx and szz
		d_sxx(gidz,gidx) += d_a_x_half[gidx]*dphi_xx_x_dx \
				+ d_ave_Byc_b(gidz, gidx)*dvx_dx/d_K_x_half[gidx]*dt;;
		d_szz(gidz,gidx) += d_a_z_half[gidz]*dphi_zz_z_dz \
				+ d_ave_Byc_a(gidz, gidx)*dvz_dz/d_K_z_half[gidz]*dt;

		// update psi_xx and psi_zz
		// if(gidx<nPml || gidx>nx-nPml-1){
			d_mem_dvx_dx(gidz, gidx) = d_b_x[gidx]*d_mem_dvx_dx(gidz, gidx) + lambda*d_szz(gidz, gidx)*dt \
				+ (lambda+2.0*mu)*d_sxx(gidz,gidx)*dt;
		// }
		// if(gidz<nPml || (gidz>nz-nPml-nPad-1)){
			d_mem_dvz_dz(gidz, gidx) = d_b_z[gidz]*d_mem_dvz_dz(gidz, gidx) + (lambda+2.0*mu)*d_szz(gidz, gidx)*dt \
				+ lambda*d_sxx(gidz,gidx)*dt;
		// }

	}

	else {
		return;
	}


}
