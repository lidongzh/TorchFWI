#include "Model.h"
#include "Cpml.h"
#include "Parameter.h"
#include "utilities.h"


Cpml::Cpml(Parameter &para, Model &model) {

	int nz = model.nz();
	int nx = model.nx();
	int nPml = para.nPoints_pml();
	int nPad = para.nPad();
	float f0 = para.f0();
	float dt = para.dt();
	float dz = para.dz();
	float dx = para.dx();

	float CpAve = compCpAve(model.h_Cp, nz*nx);

	// K_z = (float*)malloc(nz*sizeof(float));
	// a_z = (float*)malloc(nz*sizeof(float));
	// b_z = (float*)malloc(nz*sizeof(float));
	// K_z_half = (float*)malloc(nz*sizeof(float));
	// a_z_half = (float*)malloc(nz*sizeof(float));
	// b_z_half = (float*)malloc(nz*sizeof(float));

	// for padding
	K_z = (float*)malloc((nz-nPad)*sizeof(float));
	a_z = (float*)malloc((nz-nPad)*sizeof(float));
	b_z = (float*)malloc((nz-nPad)*sizeof(float));
	K_z_half = (float*)malloc((nz-nPad)*sizeof(float));
	a_z_half = (float*)malloc((nz-nPad)*sizeof(float));
	b_z_half = (float*)malloc((nz-nPad)*sizeof(float));

	K_x = (float*)malloc(nx*sizeof(float));
	a_x = (float*)malloc(nx*sizeof(float));
	b_x = (float*)malloc(nx*sizeof(float));
	K_x_half = (float*)malloc(nx*sizeof(float));
	a_x_half = (float*)malloc(nx*sizeof(float));
	b_x_half = (float*)malloc(nx*sizeof(float));

	// cpmlInit(K_z, a_z, b_z, K_z_half, \
	// a_z_half, b_z_half, nz, nPml, dz, \
	// f0, dt, CpAve);

	cpmlInit(K_z, a_z, b_z, K_z_half, \
			a_z_half, b_z_half, nz-nPad, nPml, dz, \
			f0, dt, CpAve);

	cpmlInit(K_x, a_x, b_x, K_x_half, \
	a_x_half, b_x_half, nx, nPml, dx, \
	f0, dt, CpAve);


	// allocate cpml parameters on GPU
	// CHECK(cudaMalloc((void**)&d_K_z, nz *sizeof(float)));
	// CHECK(cudaMalloc((void**)&d_a_z, nz *sizeof(float)));
	// CHECK(cudaMalloc((void**)&d_b_z, nz *sizeof(float)));
	// CHECK(cudaMalloc((void**)&d_K_z_half, nz *sizeof(float)));
	// CHECK(cudaMalloc((void**)&d_a_z_half, nz *sizeof(float)));
	// CHECK(cudaMalloc((void**)&d_b_z_half, nz *sizeof(float)));

	// for padding
	CHECK(cudaMalloc((void**)&d_K_z, (nz-nPad) *sizeof(float)));
	CHECK(cudaMalloc((void**)&d_a_z, (nz-nPad) *sizeof(float)));
	CHECK(cudaMalloc((void**)&d_b_z, (nz-nPad) *sizeof(float)));
	CHECK(cudaMalloc((void**)&d_K_z_half, (nz-nPad) *sizeof(float)));
	CHECK(cudaMalloc((void**)&d_a_z_half, (nz-nPad) *sizeof(float)));
	CHECK(cudaMalloc((void**)&d_b_z_half, (nz-nPad) *sizeof(float)));

	CHECK(cudaMalloc((void**)&d_K_x, nx *sizeof(float)));
	CHECK(cudaMalloc((void**)&d_a_x, nx *sizeof(float)));
	CHECK(cudaMalloc((void**)&d_b_x, nx *sizeof(float)));
	CHECK(cudaMalloc((void**)&d_K_x_half, nx *sizeof(float)));
	CHECK(cudaMalloc((void**)&d_a_x_half, nx *sizeof(float)));
	CHECK(cudaMalloc((void**)&d_b_x_half, nx *sizeof(float)));

	// copy cpml parameters to GPU
	// CHECK(cudaMemcpy(d_K_z, K_z, nz*sizeof(float), cudaMemcpyHostToDevice));
	// CHECK(cudaMemcpy(d_a_z, a_z, nz*sizeof(float), cudaMemcpyHostToDevice));
	// CHECK(cudaMemcpy(d_b_z, b_z, nz*sizeof(float), cudaMemcpyHostToDevice));
	// CHECK(cudaMemcpy(d_K_z_half, K_z_half, nz*sizeof(float), cudaMemcpyHostToDevice));
	// CHECK(cudaMemcpy(d_a_z_half, a_z_half, nz*sizeof(float), cudaMemcpyHostToDevice));
	// CHECK(cudaMemcpy(d_b_z_half, b_z_half, nz*sizeof(float), cudaMemcpyHostToDevice));

	// for padding
	CHECK(cudaMemcpy(d_K_z, K_z, (nz-nPad)*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_a_z, a_z, (nz-nPad)*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_b_z, b_z, (nz-nPad)*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_K_z_half, K_z_half, (nz-nPad)*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_a_z_half, a_z_half, (nz-nPad)*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_b_z_half, b_z_half, (nz-nPad)*sizeof(float), cudaMemcpyHostToDevice));

	CHECK(cudaMemcpy(d_K_x, K_x, nx*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_a_x, a_x, nx*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_b_x, b_x, nx*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_K_x_half, K_x_half, nx*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_a_x_half, a_x_half, nx*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_b_x_half, b_x_half, nx*sizeof(float), cudaMemcpyHostToDevice));

	// 	// turn off pml
	// intialArrayGPU<<<(nz-nPad+31)/32,32>>>(d_K_z, nz-nPad, 1, 1.0);
	// intialArrayGPU<<<(nz-nPad+31)/32,32>>>(d_a_z, nz-nPad, 1, 0.0);
	// intialArrayGPU<<<(nz-nPad+31)/32,32>>>(d_b_z, nz-nPad, 1, 0.0);
	// intialArrayGPU<<<(nz-nPad+31)/32,32>>>(d_K_z_half, nz-nPad, 1, 1.0);
	// intialArrayGPU<<<(nz-nPad+31)/32,32>>>(d_a_z_half, nz-nPad, 1, 0.0);
	// intialArrayGPU<<<(nz-nPad+31)/32,32>>>(d_b_z_half, nz-nPad, 1, 0.0);

	// intialArrayGPU<<<(nx+31)/32,32>>>(d_K_x, nx, 1, 1.0);
	// intialArrayGPU<<<(nx+31)/32,32>>>(d_a_x, nx, 1, 0.0);
	// intialArrayGPU<<<(nx+31)/32,32>>>(d_b_x, nx, 1, 0.0);
	// intialArrayGPU<<<(nx+31)/32,32>>>(d_K_x_half, nx, 1, 1.0);
	// intialArrayGPU<<<(nx+31)/32,32>>>(d_a_x_half, nx, 1, 0.0);
	// intialArrayGPU<<<(nx+31)/32,32>>>(d_b_x_half, nx, 1, 0.0);


}


Cpml::~Cpml() {
	free(K_z);
	free(a_z);
	free(b_z);
	free(K_z_half);
	free(a_z_half);
	free(b_z_half);
	free(K_x);
	free(a_x);
	free(b_x);
	free(K_x_half);
	free(a_x_half);
	free(b_x_half);

	CHECK(cudaFree(d_K_z));
	CHECK(cudaFree(d_a_z));
	CHECK(cudaFree(d_b_z));
	CHECK(cudaFree(d_K_z_half));
	CHECK(cudaFree(d_a_z_half));
	CHECK(cudaFree(d_b_z_half));
	CHECK(cudaFree(d_K_x));
	CHECK(cudaFree(d_a_x));
	CHECK(cudaFree(d_b_x));
	CHECK(cudaFree(d_K_x_half));
	CHECK(cudaFree(d_a_x_half));
	CHECK(cudaFree(d_b_x_half));
}