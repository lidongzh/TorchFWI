#define d_vx(z,x)  d_vx[(x)*(nz)+(z)]
#define d_vy(z,x)  d_vy[(x)*(nz)+(z)]
#define d_vz(z,x)  d_vz[(x)*(nz)+(z)]
#define d_szz(z,x) d_szz[(x)*(nz)+(z)] // Pressure
#define d_mem_dvz_dz(z,x) d_mem_dvz_dz[(x)*(nz)+(z)]
#define d_mem_dvx_dx(z,x) d_mem_dvx_dx[(x)*(nz)+(z)]
#define d_Lambda(z,x)     d_Lambda[(x)*(nz)+(z)]
#define d_Den(z,x)        d_Den[(x)*(nz)+(z)]
#define d_mat_dvz_dz(z,x) d_mat_dvz_dz[(x)*(nz)+(z)]
#define d_mat_dvx_dx(z,x) d_mat_dvx_dx[(x)*(nz)+(z)]

__global__ void ac_pressure(float *d_vz, float *d_vx, float *d_szz, \
	float *d_mem_dvz_dz, float *d_mem_dvx_dx, float *d_Lambda, \
	float *d_Den, float *d_K_z_half, float *d_a_z_half, float *d_b_z_half, \
	float *d_K_x, float *d_a_x, float *d_b_x, \
	int nz, int nx, float dt, float dz, float dx, int nPml, int nPad, bool isFor, \
	float *d_mat_dvz_dz, float *d_mat_dvx_dx){

  int gidz = blockIdx.x*blockDim.x + threadIdx.x;
  int gidx = blockIdx.y*blockDim.y + threadIdx.y;


  float dvz_dz = 0.0;
  float dvx_dx = 0.0;

  float c1 = 9.0/8.0;
  float c2 = 1.0/24.0;

  if (isFor) {

		if (gidz>=2 && gidz<=nz-nPad-3 && gidx>=2 && gidx<=nx-3) {

		  dvz_dz = (c1*(d_vz(gidz+1,gidx)-d_vz(gidz,gidx)) - c2*(d_vz(gidz+2,gidx)-d_vz(gidz-1,gidx)))/dz;
		  dvx_dx = (c1*(d_vx(gidz,gidx)-d_vx(gidz,gidx-1)) - c2*(d_vx(gidz,gidx+1)-d_vx(gidz,gidx-2)))/dx;

		  if(gidz<=nPml || (gidz>=nz-nPml-nPad-1)){
			  d_mem_dvz_dz(gidz,gidx) = d_b_z_half[gidz]*d_mem_dvz_dz(gidz,gidx) + d_a_z_half[gidz]*dvz_dz;
			  dvz_dz = dvz_dz / d_K_z_half[gidz] + d_mem_dvz_dz(gidz,gidx);
			}
			if(gidx<=nPml || gidx>=nx-nPml-1){
			  d_mem_dvx_dx(gidz,gidx) = d_b_x[gidx]*d_mem_dvx_dx(gidz,gidx) + d_a_x[gidx]*dvx_dx;
			  dvx_dx = dvx_dx / d_K_x[gidx] + d_mem_dvx_dx(gidz,gidx);
			}

		  d_szz(gidz,gidx) += d_Lambda(gidz,gidx) * (dvz_dz + dvx_dx) * dt;

		}

		else {
			return;
		}

	}

	else {
		// extension for derivative at the boundaries
		if (gidz>=nPml+2 && gidz<=nz-nPad-3-nPml && gidx>=nPml+2 && gidx<=nx-3-nPml) {
		// if (gidz>=nPml-2 && gidz<=nz-nPad+1-nPml && gidx>=nPml-2 && gidx<=nx+1-nPml) {
		  dvz_dz = (c1*(d_vz(gidz+1,gidx)-d_vz(gidz,gidx)) - c2*(d_vz(gidz+2,gidx)-d_vz(gidz-1,gidx)))/dz;
		  dvx_dx = (c1*(d_vx(gidz,gidx)-d_vx(gidz,gidx-1)) - c2*(d_vx(gidz,gidx+1)-d_vx(gidz,gidx-2)))/dx;
		  d_mat_dvz_dz(gidz, gidx) = dvz_dz;
		  d_mat_dvx_dx(gidz, gidx) = dvx_dx;

		  d_szz(gidz,gidx) -= d_Lambda(gidz,gidx) * (dvz_dz + dvx_dx) * dt;
		  // // compute the derivative at the boundaries
		  // if (gidz == nPml+2) {
		  // 	d_mat_dvz_dz(gidz-1, gidx) = 0.5 * (d_vz(gidz,gidx)-d_vz(gidz-1,gidx))/dz;
		  // 	d_mat_dvz_dz(gidz-2, gidx) = 0.5 * (d_vz(gidz-1,gidx)-d_vz(gidz-2,gidx))/dz;
		  // }
		 	// if (gidz == nz-nPad-3-nPml) {
		  // 	d_mat_dvz_dz(gidz+1, gidx) = 0.5 * (d_vz(gidz+2,gidx)-d_vz(gidz+1,gidx))/dz;
		  // 	d_mat_dvz_dz(gidz+2, gidx) = 0.5 * (d_vz(gidz+2,gidx)-d_vz(gidz+1,gidx))/dz;
		  // }

		  // if (gidx == nPml+2) {
		  // 	d_mat_dvx_dx(gidz, gidx-1) = 0.5 * (d_vx(gidz,gidx-1)-d_vx(gidz,gidx-2))/dx;
		  // 	d_mat_dvx_dx(gidz, gidx-2) = 0.5 * (d_vx(gidz,gidx-1)-d_vx(gidz,gidx-2))/dx;
		  // }
		 	// if (gidx == nx-3-nPml) {
		  // 	d_mat_dvx_dx(gidz, gidx+1) = 0.5 * (d_vx(gidz,gidx+1)-d_vx(gidz,gidx))/dx;
		  // 	d_mat_dvx_dx(gidz, gidx+2) = 0.5 * (d_vx(gidz,gidx+2)-d_vx(gidz,gidx+1))/dx;
		  // }

		}

		else {
			return;
		}
	}

}
