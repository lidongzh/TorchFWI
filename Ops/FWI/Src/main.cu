// Dongzhuo Li 05/06/2018
#include "Parameter.h"
#include "Model.h"
#include "Cpml.h"
#include "utilities.h"
#include "Src_Rec.h"
#include "Boundary.h"
#include "utilities.h"
#include <chrono>

int main(int argc, char* argv[]) {

auto start0 = std::chrono::high_resolution_clock::now();

	std::string para_fname;
	std::string survey_fname;
	if (argc != 3){
		std::cout << "Wrong input file numbers!" << std::endl;
		exit(1);
	}else{
		para_fname = std::string(argv[1]);
  	survey_fname = std::string(argv[2]);
  	std::cout << "para_fname = " << para_fname << std::endl;
	}

	// read parameter file
	Parameter para(para_fname);
	Model model(para);
	// Model model;
	Cpml cpml(para, model);
	Bnd boundaries(para);

auto startSrc = std::chrono::high_resolution_clock::now();
	Src_Rec src_rec(para, survey_fname);
auto finishSrc = std::chrono::high_resolution_clock::now(); 
std::chrono::duration<double> elapsedSrc = finishSrc - startSrc;
	std::cout << "Src_Rec time: "<< elapsedSrc.count() <<" second(s)"<< std::endl;
	std::cout << "number of shots " << src_rec.d_vec_z_rec.size() << std::endl;
	std::cout << "number of d_data " << src_rec.d_vec_data.size() << std::endl;

	// displayArray("b_z", cpml.b_x_half, 33, 1);

	int nz = model.nz();
	int nx = model.nx();
	int nPml = para.nPoints_pml();
	int nPad = para.nPad();
	float dz = para.dz();
	float dx = para.dx();
	float dt = para.dt();
	float f0 = para.f0();

	int iSnap = 0; //400
	int nrec = 1;
	float win_ratio = 0.1;
	int nSteps = para.nSteps();
	int nShots = src_rec.vec_z_src.size();

	float amp_ratio = 1.0;

	// compute Courant number
	compCourantNumber(model.h_Cp, nz*nx, dt, dz, dx);

	dim3 threads(TX,TY);
	dim3 blocks((nz+TX-1)/TX, (nx+TY-1)/TY);
	dim3 threads2(TX+4,TY+4);
	dim3 blocks2((nz+TX+3)/(TX+4), (nx+TY+3)/(TY+4));

	float *d_vz, *d_vx, *d_szz, *d_sxx, *d_sxz, *d_vz_adj, *d_vx_adj, *d_szz_adj, *d_szz_p1;
	float *d_mem_dvz_dz, *d_mem_dvz_dx, *d_mem_dvx_dz, *d_mem_dvx_dx;
	float *d_mem_dszz_dz, *d_mem_dsxx_dx, *d_mem_dsxz_dz, *d_mem_dsxz_dx;
	float *d_mat_dvz_dz, *d_mat_dvx_dx;
	float *d_l2Obj_temp;
	float *h_l2Obj_temp = nullptr;
	h_l2Obj_temp = (float*)malloc(sizeof(float));
	float h_l2Obj = 0.0;
	// float h_l2Obj_cpu = 0.0;
	CHECK(cudaMalloc((void**)&d_vz, nz * nx * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_vx, nz * nx * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_szz, nz * nx * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_sxx, nz * nx * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_sxz, nz * nx * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_vz_adj, nz * nx * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_vx_adj, nz * nx * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_szz_adj, nz * nx * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_szz_p1, nz * nx * sizeof(float)));

	CHECK(cudaMalloc((void**)&d_mem_dvz_dz, nz * nx * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_mem_dvz_dx, nz * nx * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_mem_dvx_dz, nz * nx * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_mem_dvx_dx, nz * nx * sizeof(float)));

	CHECK(cudaMalloc((void**)&d_mem_dszz_dz, nz * nx * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_mem_dsxx_dx, nz * nx * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_mem_dsxz_dz, nz * nx * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_mem_dsxz_dx, nz * nx * sizeof(float)));
	// spatial derivatives: for kernel computations
	CHECK(cudaMalloc((void**)&d_mat_dvz_dz, nz * nx * sizeof(float)));
	CHECK(cudaMalloc((void**)&d_mat_dvx_dx, nz * nx * sizeof(float)));

	CHECK(cudaMalloc((void**)&d_l2Obj_temp, 1 * sizeof(float)));


	// constant memory coefficients
	// const float h_coef[] = {9.0/8.0, 1.0/24.0};
	// cudaMemcpyToSymbol(coef, h_coef, 2*sizeof(float));

	// cudaFuncSetCacheConfig(el_stress, cudaFuncCachePreferL1);
	// cudaFuncSetCacheConfig(el_velocity, cudaFuncCachePreferL1);
	// cudaFuncSetCacheConfig(ac_pressure, cudaFuncCachePreferL1);
	// cudaFuncSetCacheConfig(ac_velocity, cudaFuncCachePreferL1);
	// cudaFuncSetCacheConfig(ac_pressure_adj, cudaFuncCachePreferL1);
	// cudaFuncSetCacheConfig(ac_velocity_adj, cudaFuncCachePreferL1);	


	float *h_snap, *h_snap_back, *h_snap_adj;
	h_snap = (float*)malloc(nz*nx*sizeof(float));
	h_snap_back = (float*)malloc(nz*nx*sizeof(float));
	h_snap_adj = (float*)malloc(nz*nx*sizeof(float));

	cudaStream_t streams[nShots];

	auto finish0 = std::chrono::high_resolution_clock::now(); 
	std::chrono::duration<double> elapsed0 = finish0 - start0;
	std::cout << "Initialization time: "<< elapsed0.count() <<" second(s)"<< std::endl;


	auto start = std::chrono::high_resolution_clock::now();

	for(int iShot=0; iShot<nShots; iShot++) {
		printf("	Processing shot %d\n", iShot);
		CHECK(cudaStreamCreate(&streams[iShot]));

// load precomputed presure DL
		// fileBinLoad(h_snap, nz*nx, "Pressure.bin");
		// CHECK(cudaMemcpy(d_szz, h_snap, nz*nx*sizeof(float), cudaMemcpyHostToDevice));
		// CHECK(cudaMemcpy(d_vx, h_snap, nz*nx*sizeof(float), cudaMemcpyHostToDevice));
		// CHECK(cudaMemcpy(d_vz, h_snap, nz*nx*sizeof(float), cudaMemcpyHostToDevice));

		intialArrayGPU<<<blocks,threads>>>(d_vz, nz, nx, 0.0);
		intialArrayGPU<<<blocks,threads>>>(d_vx, nz, nx, 0.0);
		intialArrayGPU<<<blocks,threads>>>(d_szz, nz, nx, 0.0);
		intialArrayGPU<<<blocks,threads>>>(d_sxx, nz, nx, 0.0);
		intialArrayGPU<<<blocks,threads>>>(d_sxz, nz, nx, 0.0);

		intialArrayGPU<<<blocks,threads>>>(d_mem_dvz_dz, nz, nx, 0.0);
		intialArrayGPU<<<blocks,threads>>>(d_mem_dvz_dx, nz, nx, 0.0);
		intialArrayGPU<<<blocks,threads>>>(d_mem_dvx_dz, nz, nx, 0.0);
		intialArrayGPU<<<blocks,threads>>>(d_mem_dvx_dx, nz, nx, 0.0);

		intialArrayGPU<<<blocks,threads>>>(d_mem_dszz_dz, nz, nx, 0.0);
		intialArrayGPU<<<blocks,threads>>>(d_mem_dsxx_dx, nz, nx, 0.0);
		intialArrayGPU<<<blocks,threads>>>(d_mem_dsxz_dz, nz, nx, 0.0);
		intialArrayGPU<<<blocks,threads>>>(d_mem_dsxz_dx, nz, nx, 0.0);

		intialArrayGPU<<<blocks,threads>>>(d_mat_dvz_dz, nz, nx, 0.0);
		intialArrayGPU<<<blocks,threads>>>(d_mat_dvx_dx, nz, nx, 0.0);

		nrec = src_rec.vec_nrec.at(iShot);
		if (para.if_res()) {
			fileBinLoad(src_rec.vec_data_obs.at(iShot), nSteps*nrec, para.data_dir_name() \
					+ "Shot" + std::to_string(iShot) + ".bin");
			CHECK(cudaMemcpyAsync(src_rec.d_vec_data_obs.at(iShot), \
					src_rec.vec_data_obs.at(iShot), nrec * nSteps * sizeof(float), \
					cudaMemcpyHostToDevice, streams[iShot]));
		}
	// ------------------------------------ time loop ------------------------------------
		for(int it=0; it<=nSteps-2; it++){

			// =========================== elastic or acoustic ===========================
			if (para.withAdj()) {
				// save and record from the beginning
				boundaries.field_from_bnd(d_szz, d_sxz, d_sxx, d_vz, d_vx, it);
			}

			// get snapshot at time it
			if(it==iSnap&&iShot==0){
				CHECK(cudaMemcpy(h_snap, d_szz, nz*nx*sizeof(float), cudaMemcpyDeviceToHost));
			}

			if (para.isAc()) {

				ac_pressure<<<blocks,threads>>>(d_vz, d_vx, d_szz, \
						d_mem_dvz_dz, d_mem_dvx_dx, model.d_Lambda, \
						model.d_Den, cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half, \
						cpml.d_K_x, cpml.d_a_x, cpml.d_b_x, \
						nz, nx, dt, dz, dx, nPml, nPad, true, d_mat_dvz_dz, d_mat_dvx_dx);

				add_source<<<1,1>>>(d_szz, d_sxx, src_rec.vec_source.at(iShot)[it], nz, \
						true, src_rec.vec_z_src.at(iShot), src_rec.vec_x_src.at(iShot), dt, model.d_Cp);

				ac_velocity<<<blocks,threads>>>(d_vz, d_vx, d_szz, \
						d_mem_dszz_dz, d_mem_dsxx_dx, model.d_Lambda, \
						model.d_Den, model.d_ave_Byc_a, model.d_ave_Byc_b, \
						cpml.d_K_z, cpml.d_a_z, cpml.d_b_z, \
						cpml.d_K_x_half, cpml.d_a_x_half, cpml.d_b_x_half, \
						nz, nx, dt, dz, dx, nPml, nPad, true);
			} else {

				el_stress<<<blocks,threads>>>(d_vz, d_vx, d_szz, \
						d_sxx, d_sxz, d_mem_dvz_dz, d_mem_dvz_dx, \
						d_mem_dvx_dz, d_mem_dvx_dx, model.d_Lambda, model.d_Mu, model.d_ave_Mu,\
						model.d_Den, cpml.d_K_z, cpml.d_a_z, cpml.d_b_z, cpml.d_K_z_half, \
						cpml.d_a_z_half, cpml.d_b_z_half, cpml.d_K_x, cpml.d_a_x, cpml.d_b_x, \
						cpml.d_K_x_half, cpml.d_a_x_half, cpml.d_b_x_half, \
						nz, nx, dt, dz, dx, nPml, nPad, true);

				add_source<<<1,1>>>(d_szz, d_sxx, src_rec.vec_source.at(iShot)[it], nz, \
						true, src_rec.vec_z_src.at(iShot), src_rec.vec_x_src.at(iShot), dt, model.d_Cp);

				el_velocity<<<blocks,threads>>>(d_vz, d_vx, d_szz, \
						d_sxx, d_sxz, d_mem_dszz_dz, d_mem_dsxz_dx, d_mem_dsxz_dz, \
						d_mem_dsxx_dx, model.d_Lambda, model.d_Mu, model.d_ave_Byc_a, model.d_ave_Byc_b, \
						cpml.d_K_z, cpml.d_a_z, cpml.d_b_z, \
						cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half, cpml.d_K_x, cpml.d_a_x, cpml.d_b_x, \
						cpml.d_K_x_half, cpml.d_a_x_half, cpml.d_b_x_half, nz, nx, dt, dz, dx, nPml, nPad, true);

			}
			recording<<<(nrec+31)/32,32>>>(d_szz, nz, src_rec.d_vec_data.at(iShot), iShot, it+1, nSteps, nrec, \
				src_rec.d_vec_z_rec.at(iShot), src_rec.d_vec_x_rec.at(iShot));

		}

		if (!para.if_res()) {
			CHECK(cudaMemcpyAsync(src_rec.vec_data.at(iShot), src_rec.d_vec_data.at(iShot), \
			nSteps*nrec*sizeof(float), cudaMemcpyDeviceToHost, streams[iShot])); // test
		}

		fileBinWrite(h_snap, nz*nx, "SnapGPU.bin");



		// compute residuals
		if (para.if_res()) {
			dim3 blocksT((nSteps+TX-1)/TX, (nrec+TY-1)/TY);

			// for fun modify observed data
			// float filter2[4] = {8.0, 9.0, 12.0, 13.0};
			// cuda_window<<<blocksT,threads>>>(nSteps, nrec, dt, win_ratio, src_rec.d_vec_data_obs.at(iShot));
			// bp_filter1d(nSteps, dt, nrec, src_rec.d_vec_data_obs.at(iShot), filter2);


			// // windowing
			// if (para.if_win()) {
			// 	cuda_window<<<blocksT,threads>>>(nSteps, nrec, dt, src_rec.d_vec_win_start.at(iShot), \
   //  			src_rec.d_vec_win_end.at(iShot), src_rec.d_vec_weights.at(iShot), win_ratio, src_rec.d_vec_data_obs.at(iShot));
			// 	cuda_window<<<blocksT,threads>>>(nSteps, nrec, dt, src_rec.d_vec_win_start.at(iShot), \
   //  			src_rec.d_vec_win_end.at(iShot), src_rec.d_vec_weights.at(iShot), win_ratio, src_rec.d_vec_data.at(iShot));
			// } else {
			// 	cuda_window<<<blocksT,threads>>>(nSteps, nrec, dt, win_ratio, src_rec.d_vec_data_obs.at(iShot));
			// 	cuda_window<<<blocksT,threads>>>(nSteps, nrec, dt, win_ratio, src_rec.d_vec_data.at(iShot));
			// }


			// // filtering
			// bp_filter1d(nSteps, dt, nrec, src_rec.d_vec_data_obs.at(iShot), para.filter());
			// bp_filter1d(nSteps, dt, nrec, src_rec.d_vec_data.at(iShot), para.filter());


			// // Calculate source update and filter calculated data
			// if (para.if_src_update()) {
			// 	amp_ratio = source_update(nSteps, dt, nrec, src_rec.d_vec_data_obs.at(iShot), \
			// 			src_rec.d_vec_data.at(iShot), src_rec.d_vec_source.at(iShot), src_rec.d_coef);
			// 	printf("	Source update => Processing shot %d, amp_ratio = %f\n", iShot, amp_ratio);
			// }
			// amp_ratio = 1.0; // amplitude not used, so set to 1.0

			// objective function
			gpuMinus<<<blocksT, threads>>>(src_rec.d_vec_res.at(iShot), src_rec.d_vec_data_obs.at(iShot), \
					src_rec.d_vec_data.at(iShot), nSteps, nrec);
			cuda_cal_objective<<<1,512>>>(d_l2Obj_temp, src_rec.d_vec_res.at(iShot), nSteps*nrec);
			CHECK(cudaMemcpy(h_l2Obj_temp, d_l2Obj_temp, sizeof(float), cudaMemcpyDeviceToHost));
			h_l2Obj += h_l2Obj_temp[0];


			// //  update source again (adjoint)
			// if (para.if_src_update()) {
			// 	source_update_adj(nSteps, dt, nrec, src_rec.d_vec_res.at(iShot), \
   //                     amp_ratio, src_rec.d_coef);
			// }

			// // filtering again (adjoint)
			// bp_filter1d(nSteps, dt, nrec, src_rec.d_vec_res.at(iShot), para.filter());
			// // windowing again (adjoint)
			// if (para.if_win()) {
			// 	cuda_window<<<blocksT,threads>>>(nSteps, nrec, dt, src_rec.d_vec_win_start.at(iShot), \
   //  			src_rec.d_vec_win_end.at(iShot), src_rec.d_vec_weights.at(iShot), win_ratio, src_rec.d_vec_res.at(iShot));
			// } else {
			// 	cuda_window<<<blocksT,threads>>>(nSteps, nrec, dt, win_ratio, src_rec.d_vec_res.at(iShot));
			// }

			CHECK(cudaMemcpyAsync(src_rec.vec_res.at(iShot), src_rec.d_vec_res.at(iShot), \
				nSteps*nrec*sizeof(float), cudaMemcpyDeviceToHost, streams[iShot])); // test
			// CHECK(cudaMemcpy(src_rec.vec_res.at(iShot), src_rec.d_vec_res.at(iShot), \
			// 	nSteps*nrec*sizeof(float), cudaMemcpyDeviceToHost)); // test
			CHECK(cudaMemcpyAsync(src_rec.vec_data.at(iShot), src_rec.d_vec_data.at(iShot), \
				nSteps*nrec*sizeof(float), cudaMemcpyDeviceToHost, streams[iShot])); // test
			CHECK(cudaMemcpyAsync(src_rec.vec_data_obs.at(iShot), src_rec.d_vec_data_obs.at(iShot), \
				nSteps*nrec*sizeof(float), cudaMemcpyDeviceToHost, streams[iShot])); // save preconditioned observed
			CHECK(cudaMemcpy(src_rec.vec_source.at(iShot), src_rec.d_vec_source.at(iShot), \
				nSteps*sizeof(float), cudaMemcpyDeviceToHost));
		}
		// =================
		cudaDeviceSynchronize();


		if (para.withAdj()) {
			// ------------------------------------- Backward ----------------------------------
			// initialization
			intialArrayGPU<<<blocks,threads>>>(d_vz_adj, nz, nx, 0.0);
			intialArrayGPU<<<blocks,threads>>>(d_vx_adj, nz, nx, 0.0);
			intialArrayGPU<<<blocks,threads>>>(d_szz_adj, nz, nx, 0.0);
			intialArrayGPU<<<blocks,threads>>>(d_szz_p1, nz, nx, 0.0);
			intialArrayGPU<<<blocks,threads>>>(d_mem_dvz_dz, nz, nx, 0.0);
			intialArrayGPU<<<blocks,threads>>>(d_mem_dvx_dx, nz, nx, 0.0);
			intialArrayGPU<<<blocks,threads>>>(d_mem_dszz_dz, nz, nx, 0.0);
			intialArrayGPU<<<blocks,threads>>>(d_mem_dsxx_dx, nz, nx, 0.0);

			for(int it=nSteps-2; it>=0; it--) {

				if (para.isAc()) {

					// if (it <= nSteps - 2) {
						// save p to szz_plus_one
						assignArrayGPU<<<blocks,threads>>>(d_szz, d_szz_p1, nz, nx);
						//value at T-1
						ac_velocity<<<blocks,threads>>>(d_vz, d_vx, d_szz, \
								d_mem_dszz_dz, d_mem_dsxx_dx, model.d_Lambda, \
								model.d_Den, model.d_ave_Byc_a, model.d_ave_Byc_b, \
								cpml.d_K_z, cpml.d_a_z, cpml.d_b_z, \
								cpml.d_K_x_half, cpml.d_a_x_half, cpml.d_b_x_half, \
								nz, nx, dt, dz, dx, nPml, nPad, false);
						boundaries.field_to_bnd(d_szz, d_sxz, d_sxx, d_vz, d_vx, it, false);

						add_source<<<1,1>>>(d_szz, d_sxx, src_rec.vec_source.at(iShot)[it], nz, \
								false, src_rec.vec_z_src.at(iShot), src_rec.vec_x_src.at(iShot), dt, model.d_Cp);
						add_source<<<1,1>>>(d_szz_p1, d_sxx, src_rec.vec_source.at(iShot)[it], nz, \
								false, src_rec.vec_z_src.at(iShot), src_rec.vec_x_src.at(iShot), dt, model.d_Cp);

						ac_pressure<<<blocks,threads>>>(d_vz, d_vx, d_szz, \
								d_mem_dvz_dz, d_mem_dvx_dx, model.d_Lambda, \
								model.d_Den, cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half, \
								cpml.d_K_x, cpml.d_a_x, cpml.d_b_x, \
								nz, nx, dt, dz, dx, nPml, nPad, false, d_mat_dvz_dz, d_mat_dvx_dx);

						boundaries.field_to_bnd(d_szz, d_sxz, d_sxx, d_vz, d_vx, it, true);
						// value at T-2


					// ================
					// adjoint computation

					ac_velocity_adj<<<blocks,threads>>>(d_vz_adj, d_vx_adj, d_szz_adj, \
							d_mem_dvz_dz, d_mem_dvx_dx, d_mem_dszz_dz, d_mem_dsxx_dx, \
							model.d_Lambda, model.d_Den, model.d_ave_Byc_a, model.d_ave_Byc_b, \
							cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half, \
							cpml.d_K_x_half, cpml.d_a_x_half, cpml.d_b_x_half, \
							cpml.d_K_z, cpml.d_a_z, cpml.d_b_z, \
							cpml.d_K_x, cpml.d_a_x, cpml.d_b_x, \
							nz, nx, dt, dz, dx, nPml, nPad);

					// inject residuals
					res_injection<<<(nrec+31)/32,32>>>(d_szz_adj, nz, src_rec.d_vec_res.at(iShot), \
							model.d_Lambda, it+1, dt, nSteps, nrec, src_rec.d_vec_z_rec.at(iShot), \
							src_rec.d_vec_x_rec.at(iShot));

					ac_pressure_adj<<<blocks,threads>>>(d_vz_adj, d_vx_adj, d_szz_adj, \
							d_mem_dvz_dz, d_mem_dvx_dx, d_mem_dszz_dz, d_mem_dsxx_dx, \
							model.d_Lambda, model.d_Den, model.d_ave_Byc_a, model.d_ave_Byc_b, \
							cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half, \
							cpml.d_K_x_half, cpml.d_a_x_half, cpml.d_b_x_half, \
							cpml.d_K_z, cpml.d_a_z, cpml.d_b_z, \
							cpml.d_K_x, cpml.d_a_x, cpml.d_b_x, \
							nz, nx, dt, dz, dx, nPml, nPad, \
							model.d_Cp, d_mat_dvz_dz, d_mat_dvx_dx, model.d_CpGrad);
					// value at T-1

					// ac_adj_push<<<blocks,threads2>>>(d_vz_adj, d_vx_adj, d_szz_adj, d_adj_temp, \
					// 		d_mem_dvz_dz, d_mem_dvx_dx, d_mem_dszz_dz, d_mem_dsxx_dx, \
					// 		model.d_Lambda, model.d_Den, model.d_ave_Byc_a, model.d_ave_Byc_b, \
					// 		cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half, \
					// 		cpml.d_K_x_half, cpml.d_a_x_half, cpml.d_b_x_half, \
					// 		cpml.d_K_z, cpml.d_a_z, cpml.d_b_z, \
					// 		cpml.d_K_x, cpml.d_a_x, cpml.d_b_x, \
					// 		nz, nx, dt, dz, dx, nPml, nPad);


					// image_vel<<<blocks,threads>>>(d_szz_adj, nz, nx, dt, dz, dx, nPml, nPad, \
     			//         model.d_Cp, model.d_Den, d_mat_dvz_dz, d_mat_dvx_dx, model.d_CpGrad);
					image_vel_time<<<blocks,threads>>>(d_szz, d_szz_p1, d_szz_adj, \
                 nz, nx, dt, dz, dx, nPml, nPad, model.d_Cp, model.d_Lambda, model.d_CpGrad);

				} else {

					el_velocity<<<blocks,threads>>>(d_vz, d_vx, d_szz, \
							d_sxx, d_sxz, d_mem_dszz_dz, d_mem_dsxz_dx, d_mem_dsxz_dz, \
							d_mem_dsxx_dx, model.d_Lambda, model.d_Mu, model.d_ave_Byc_a, model.d_ave_Byc_b, \
							cpml.d_K_z, cpml.d_a_z, cpml.d_b_z, \
							cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half, cpml.d_K_x, cpml.d_a_x, cpml.d_b_x, \
							cpml.d_K_x_half, cpml.d_a_x_half, cpml.d_b_x_half, nz, nx, dt, dz, dx, nPml, nPad, false);


					el_stress<<<blocks,threads>>>(d_vz, d_vx, d_szz, \
							d_sxx, d_sxz, d_mem_dvz_dz, d_mem_dvz_dx, \
							d_mem_dvx_dz, d_mem_dvx_dx, model.d_Lambda, model.d_Mu, model.d_ave_Mu,\
							model.d_Den, cpml.d_K_z, cpml.d_a_z, cpml.d_b_z, cpml.d_K_z_half, \
							cpml.d_a_z_half, cpml.d_b_z_half, cpml.d_K_x, cpml.d_a_x, cpml.d_b_x, \
							cpml.d_K_x_half, cpml.d_a_x_half, cpml.d_b_x_half, \
							nz, nx, dt, dz, dx, nPml, nPad, false);

				}

				// boundaries.field_to_bnd(d_szz, d_sxz, d_sxx, d_vz, d_vx, it);

				if(it==iSnap&&iShot==0){
					CHECK(cudaMemcpy(h_snap_back, d_szz, nz*nx*sizeof(float), cudaMemcpyDeviceToHost));
					CHECK(cudaMemcpy(h_snap_adj, d_szz_adj, nz*nx*sizeof(float), cudaMemcpyDeviceToHost));
				}
				if (iShot==0) {
					// CHECK(cudaMemcpy(h_snap_adj, d_szz_adj, nz*nx*sizeof(float), cudaMemcpyDeviceToHost));
					// fileBinWrite(h_snap_adj, nz*nx, "SnapGPU_adj_" + std::to_string(it) + ".bin");
					// CHECK(cudaMemcpy(h_snap, d_szz, nz*nx*sizeof(float), cudaMemcpyDeviceToHost));
					// fileBinWrite(h_snap, nz*nx, "SnapGPU_" + std::to_string(it) + ".bin");
				}

			}
			// fileBinWrite(h_snap_back, nz*nx, "SnapGPU_back.bin");
			// fileBinWrite(h_snap_adj, nz*nx, "SnapGPU_adj.bin");
			CHECK(cudaMemcpy(model.h_CpGrad, model.d_CpGrad, nz*nx*sizeof(float), cudaMemcpyDeviceToHost));
			fileBinWrite(model.h_CpGrad, nz*nx, "CpGradient.bin");
		}

	}

	auto finish = std::chrono::high_resolution_clock::now(); 
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Elapsed time: "<< elapsed.count() <<" second(s)."<< std::endl;

	// cudaDeviceSynchronize();

	if (!para.if_res()) {
		for(int iShot = 0; iShot < nShots; iShot++){
			fileBinWrite(src_rec.vec_data.at(iShot), nSteps*src_rec.vec_nrec.at(iShot), para.data_dir_name() \
					+ "Shot" + std::to_string(iShot) + ".bin");
		}
	}
	else {
		for(int iShot = 0; iShot < nShots; iShot++){
			fileBinWrite(src_rec.vec_res.at(iShot), nSteps*src_rec.vec_nrec.at(iShot), para.data_dir_name() \
					+ "Residual_Shot" + std::to_string(iShot) + ".bin");
			fileBinWrite(src_rec.vec_data.at(iShot), nSteps*src_rec.vec_nrec.at(iShot), para.data_dir_name() \
					+ "Syn_Shot" + std::to_string(iShot) + ".bin");
			fileBinWrite(src_rec.vec_data_obs.at(iShot), nSteps*src_rec.vec_nrec.at(iShot), para.data_dir_name() \
				+ "CondObs_Shot" + std::to_string(iShot) + ".bin");
			fileBinWrite(src_rec.vec_source.at(iShot), nSteps, para.data_dir_name()\
				+ "src_updated" + std::to_string(iShot) + ".bin");
		}
	}


	// if (para.if_res()) {
	// 	// test cpu and gpu residual calculations
	// 	float h_l2Obj_cpu = cal_objective(src_rec.vec_res.at(0), nSteps*nrec);
	// 	std::cout << "cpu misfit = " << std::to_string(h_l2Obj_cpu) << std::endl;
	// }

	//output residual
	if (para.if_res()) {
		std::cout << "Total l2 residual = " << std::to_string(h_l2Obj) << std::endl;
		// std::cout << "Total l2 residual cpu = " << h_l2Obj_cpu << std::endl;
		h_l2Obj = 0.5 * h_l2Obj; // DL 02/21/2019 (need to make misfit accurate here rather than in the script)
		// fileBinWrite(&h_l2Obj, 1, "l2Obj.bin");
		fileBinWrite(&h_l2Obj, 1, "l2Obj.bin");
	}
	free(h_l2Obj_temp);


	free(h_snap);
	free(h_snap_back);
	free(h_snap_adj);

	// destroy the streams
	for (int iShot = 0; iShot < nShots; iShot++)
   	CHECK(cudaStreamDestroy(streams[iShot]));


	cudaFree(d_vz); cudaFree(d_vx); cudaFree(d_szz); cudaFree(d_sxx); cudaFree(d_sxz);
	cudaFree(d_vz_adj); cudaFree(d_vx_adj); cudaFree(d_szz_adj); cudaFree(d_szz_p1);
	cudaFree(d_mem_dvz_dz); cudaFree(d_mem_dvz_dx);
	cudaFree(d_mem_dvx_dz); cudaFree(d_mem_dvx_dx);
	cudaFree(d_mem_dszz_dz); cudaFree(d_mem_dsxx_dx);
	cudaFree(d_mem_dsxz_dz); cudaFree(d_mem_dsxz_dx);
	cudaFree(d_mat_dvz_dz); cudaFree(d_mat_dvx_dx);
	cudaFree(d_l2Obj_temp);


	return 0;
}