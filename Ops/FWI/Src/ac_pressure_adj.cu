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
#define d_mat_dvz_dz(z,x) d_mat_dvz_dz[(x)*(nz)+(z)]
#define d_mat_dvx_dx(z,x) d_mat_dvx_dx[(x)*(nz)+(z)]
#define d_Cp(z,x)        	d_Cp[(x)*(nz)+(z)]
#define d_CpGrad(z,x)		d_CpGrad[(x)*(nz)+(z)]


__global__ void ac_pressure_adj(float *d_vz, float *d_vx, float *d_szz, \
	float *d_mem_dvz_dz, float *d_mem_dvx_dx, float *d_mem_dszz_dz, float *d_mem_dsxx_dx, \
	float *d_Lambda, float *d_Den, float *d_ave_Byc_a, float *d_ave_Byc_b,\
	float *d_K_z_half, float *d_a_z_half, float *d_b_z_half, \
	float *d_K_x_half, float *d_a_x_half, float *d_b_x_half, \
	float *d_K_z, float *d_a_z, float *d_b_z, \
	float *d_K_x, float *d_a_x, float *d_b_x, \
	int nz, int nx, float dt, float dz, float dx, int nPml, int nPad, \
	float *d_Cp, float *d_mat_dvz_dz, float *d_mat_dvx_dx, float * d_CpGrad){

  int gidz = blockIdx.x*blockDim.x + threadIdx.x;
  int gidx = blockIdx.y*blockDim.y + threadIdx.y;


  float dvz_dz = 0.0;
  float dvx_dx = 0.0;
  float dphiz_dz = 0.0;
  float dphix_dx = 0.0;

  float c1 = 9.0/8.0;
  float c2 = 1.0/24.0;



	if (gidz>=2 && gidz<=nz-nPad-3 && gidx>=2 && gidx<=nx-3) {

	 //  dvz_dz = c1*(d_vz(gidz+1,gidx)-d_vz(gidz,gidx)) - c2*(d_vz(gidz+2,gidx)-d_vz(gidz-1,gidx));
	 //  dvx_dx = c1*(d_vx(gidz,gidx)-d_vx(gidz,gidx-1)) - c2*(d_vx(gidz,gidx+1)-d_vx(gidz,gidx-2));
	 //  dphiz_dz = c1*(d_mem_dszz_dz(gidz+1,gidx)-d_mem_dszz_dz(gidz,gidx)) \
	 //  		- c2*(d_mem_dszz_dz(gidz+2,gidx)-d_mem_dszz_dz(gidz-1,gidx));
	 //  dphix_dx = c1*(d_mem_dsxx_dx(gidz,gidx)-d_mem_dsxx_dx(gidz,gidx-1)) \
	 //  		- c2*(d_mem_dsxx_dx(gidz,gidx+1)-d_mem_dsxx_dx(gidz,gidx-2));

	 //  // update stress
		// d_szz(gidz,gidx) += -1.0 * d_Lambda(gidz,gidx)*dt * (d_a_x_half[gidx]*dphix_dx + d_a_z[gidz]*dphiz_dz \
		// 		+ dvx_dx/d_K_x_half[gidx]/dx + dvz_dz/d_K_z[gidz]/dz);

		// d_mem_dvx_dx(gidz, gidx) = d_b_x[gidx]*d_mem_dvx_dx(gidz, gidx) + d_szz(gidz, gidx)/dx;
		// d_mem_dvz_dz(gidz, gidx) = d_b_z_half[gidz]*d_mem_dvz_dz(gidz, gidx) + d_szz(gidz, gidx)/dz;

		// forward difference

		// if (gidz == 2) {
		//   dvz_dz = c1*(d_vz(2,gidx)-d_vz(3,gidx)) + c2*d_vz(4,gidx);
		//   dphiz_dz = c1*(d_mem_dszz_dz(2,gidx)-d_mem_dszz_dz(3,gidx)) + c2*d_mem_dszz_dz(4,gidx);			
		// } 
		// else if (gidz == nz-nPad-3) {
		//   dvz_dz = c1*d_vz(gidz,gidx) - c2*d_vz(gidz-1,gidx);
		//   dphiz_dz = c1*d_mem_dszz_dz(gidz,gidx) - c2*d_mem_dszz_dz(gidz-1,gidx);			
		// }
		// else if (gidz == nz-nPad-4) {
		//   dvz_dz = -c1*(d_vz(gidz+1,gidx)-d_vz(gidz,gidx)) - c2*d_vz(gidz-1,gidx);
		//   dphiz_dz = -c1*(d_mem_dszz_dz(gidz+1,gidx)-d_mem_dszz_dz(gidz,gidx)) - c2*d_mem_dszz_dz(gidz-1,gidx);						
		// }
		// else {
		  dvz_dz = (-c1*(d_vz(gidz+1,gidx)-d_vz(gidz,gidx)) + c2*(d_vz(gidz+2,gidx)-d_vz(gidz-1,gidx)))/dz;
		  dphiz_dz = (-c1*(d_mem_dszz_dz(gidz+1,gidx)-d_mem_dszz_dz(gidz,gidx)) \
		  		+ c2*(d_mem_dszz_dz(gidz+2,gidx)-d_mem_dszz_dz(gidz-1,gidx)))/dz;			
		// }

		// backward difference
		// if (gidx == 2) {
		//   dvx_dx = -c1*d_vx(gidz,gidx) + c2*d_vx(gidz,gidx+1);
		//   dphix_dx = -c1*d_mem_dsxx_dx(gidz,gidx) + c2*d_mem_dsxx_dx(gidz,gidx+1);			
		// }
		// if (gidx == 3) {
		//   dvx_dx = -c1*(d_vx(gidz,gidx)-d_vx(gidz,gidx-1)) + c2*d_vx(gidz,gidx+1);
		//   dphix_dx = -c1*(d_mem_dsxx_dx(gidz,gidx)-d_mem_dsxx_dx(gidz,gidx-1)) \
		//   		+ c2*d_mem_dsxx_dx(gidz,gidx+1);				
		// }
		// else if (gidx == nx-3) {
		//   dvx_dx = c1*(d_vx(gidz,gidx-1)-d_vx(gidz,gidx)) - c2*d_vx(gidz,gidx-2);
		//   dphix_dx = c1*(d_mem_dsxx_dx(gidz,gidx-1)-d_mem_dsxx_dx(gidz,gidx)) - c2*d_mem_dsxx_dx(gidz,gidx-2);						
		// } 
		// else {
		  dvx_dx = (-c1*(d_vx(gidz,gidx)-d_vx(gidz,gidx-1)) + c2*(d_vx(gidz,gidx+1)-d_vx(gidz,gidx-2)))/dx;
		  dphix_dx = (-c1*(d_mem_dsxx_dx(gidz,gidx)-d_mem_dsxx_dx(gidz,gidx-1)) \
		  		+ c2*(d_mem_dsxx_dx(gidz,gidx+1)-d_mem_dsxx_dx(gidz,gidx-2)))/dx;			
		// }

	  // update stress
		d_szz(gidz,gidx) += d_a_x_half[gidx]*dphix_dx + d_a_z[gidz]*dphiz_dz \
				+ d_ave_Byc_b(gidz, gidx)*dvx_dx/d_K_x_half[gidx]*dt + d_ave_Byc_a(gidz, gidx)*dvz_dz/d_K_z[gidz]*dt;

		if(gidx<=nPml || gidx>=nx-nPml-1){
			d_mem_dvx_dx(gidz, gidx) = d_b_x[gidx]*d_mem_dvx_dx(gidz, gidx) + d_Lambda(gidz, gidx)*d_szz(gidz, gidx)*dt;
		}
		if(gidz<=nPml || (gidz>=nz-nPml-nPad-1)){
			d_mem_dvz_dz(gidz, gidx) = d_b_z_half[gidz]*d_mem_dvz_dz(gidz, gidx) + d_Lambda(gidz, gidx)*d_szz(gidz, gidx)*dt;
		}

	}

	else {
		return;
	}


}
