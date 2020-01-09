#define d_vx(z,x)  d_vx[(x)*(nz)+(z)]
#define d_vy(z,x)  d_vy[(x)*(nz)+(z)]
#define d_vz(z,x)  d_vz[(x)*(nz)+(z)]
#define d_szz(z,x) d_szz[(x)*(nz)+(z)] // Pressure
#define d_mem_dvz_dz(z,x) d_mem_dvz_dz[(x)*(nz)+(z)]
#define d_mem_dvx_dx(z,x) d_mem_dvx_dx[(x)*(nz)+(z)]
#define d_Lambda(z,x)     d_Lambda[(x)*(nz)+(z)]
#define d_Den(z,x)        d_Den[(x)*(nz)+(z)]
#define d_Cp(z,x)        	d_Cp[(x)*(nz)+(z)]
#define d_CpGrad(z,x)		d_CpGrad[(x)*(nz)+(z)]
#define d_szz_plusone(z, x) d_szz_plusone[(x)*(nz)+(z)]
#define d_szz_adj(z, x)   d_szz_adj[(x)*(nz)+(z)]


__global__ void image_vel_time(float *d_szz, float *d_szz_plusone, float *d_szz_adj,\
	int nz, int nx, float dt, float dz, float dx, int nPml, int nPad, \
	float *d_Cp, float *d_Lambda, float *d_CpGrad){

  int gidz = blockIdx.x*blockDim.x + threadIdx.x;
  int gidx = blockIdx.y*blockDim.y + threadIdx.y;


	// if (gidz>=2 && gidz<=nz-nPad-3 && gidx>=2 && gidx<=nx-3) {
	if (gidz>=nPml && gidz<=nz-nPml-nPad-1 && gidx>=nPml && gidx<=nx-nPml-1) {

	  // compute the Vp kernel on the fly
		// d_szz_plusone(gidz, gidx) = (d_szz_plusone(gidz, gidx) - d_szz(gidz, gidx))/dt;

	 //  d_CpGrad(gidz, gidx) += -2.0 / d_Cp(gidz, gidx) / d_Lambda(gidz, gidx) \
	 //  		* d_szz_plusone(gidz, gidx) * d_szz_adj(gidz, gidx);


		d_szz_plusone(gidz, gidx) = (d_szz_plusone(gidz, gidx) - d_szz(gidz, gidx));
	  d_CpGrad(gidz, gidx) += -2.0 / d_Cp(gidz, gidx) \
	  		* d_szz_plusone(gidz, gidx) * d_szz_adj(gidz, gidx);

	}

	else {
		return;
	}


}