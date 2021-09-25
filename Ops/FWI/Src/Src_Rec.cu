// Dongzhuo Li 05/13/2018
#include <iostream>
#include <fstream>
#include <string>
#include "Src_Rec.h"
#include "utilities.h"
#include <cuda_runtime.h>

#define stf(iShot,it) stf[(iShot)*(nSteps)+(it)] // row-major
// #define VERBOSE

using namespace std;
using namespace rapidjson;

Src_Rec::Src_Rec() {
	cout << "ERROR: You need to input parameter!" << endl;
	exit(1);
}

Src_Rec::Src_Rec(Parameter &para, string survey_fname, const float *stf, int group_size, const int *shot_ids) {

	string line;
	ifstream src_rec_file;

	src_rec_file.open(survey_fname);

	if (!src_rec_file.is_open()) {
		cout << "Error opening survey file" << endl;
		exit(1);
	}

	getline(src_rec_file, line);
	src_rec_file.close();

	Document json_src_rec;
	json_src_rec.Parse<0>(line.c_str());
	assert(json_src_rec.IsObject());

	// int nsrc = 0; // for multiple sources per shot (future)
	// int *h_z_src = nullptr;
	// int *h_x_src = nullptr;
	int nrec = 0;
	int z_src = 10;
	int x_src = 10;
	int nSteps = para.nSteps();
	int nPml = para.nPoints_pml();
	float dt = para.dt();
	int *h_z_rec = nullptr;
	int *h_x_rec = nullptr;
	float *h_win_start = nullptr; // for selected window
	float *h_win_end = nullptr; // for selected window
	float *h_weights = nullptr; // for trace weighting
	float *h_source = nullptr;
	float *h_data = nullptr;
	float *h_data_obs = nullptr;
	float *h_res = nullptr;
	double *h_rec_rxz = nullptr; // for rec sxx/szz ratio
	char thisShot[10]; // for shot number small than 99999
	int *d_z_rec, *d_x_rec;
	// float *d_data;
	// float *d_data_obs;
	// float *d_res;
	float *d_source;
	float *d_win_start, *d_win_end;
	float *d_weights;

	// dim3 threads(TX,TY);
	// dim3 blocks((para.nz()+TX-1)/TX, (para.nx()+TY-1)/TY);

	if_res_ = para.if_res();
	if_win_ = para.if_win();

	assert(json_src_rec.HasMember("nShots"));
	assert(json_src_rec["nShots"].IsInt());
	nShots = json_src_rec["nShots"].GetInt();
#ifdef VERBOSE
	cout << "	nShots = " << nShots << endl;
#endif

	CHECK(cudaMalloc((void **)&d_coef, (nSteps+1) * sizeof(cuFloatComplex)));

	for (int i = 0; i < group_size; i++) {

		// get the source positions
#ifdef VERBOSE
		std::cout << "shot_id = " << shot_ids[i] << std::endl;
#endif
		strcpy(thisShot, ("shot" + to_string(shot_ids[i])).c_str());
		assert(json_src_rec[thisShot].HasMember("z_src"));
		assert(json_src_rec[thisShot]["z_src"].IsInt());
		z_src = json_src_rec[thisShot]["z_src"].GetInt() + nPml;
		vec_z_src.push_back(z_src);
		assert(json_src_rec[thisShot].HasMember("x_src"));
		assert(json_src_rec[thisShot]["x_src"].IsInt());
		x_src = json_src_rec[thisShot]["x_src"].GetInt() + nPml;
		vec_x_src.push_back(x_src);

		assert(json_src_rec[thisShot].HasMember("nrec"));
		assert(json_src_rec[thisShot]["nrec"].IsInt());
		nrec = json_src_rec[thisShot]["nrec"].GetInt();
		vec_nrec.push_back(nrec);// get the number of rec for each shot
		h_z_rec = new int[nrec];
		h_x_rec = new int[nrec];

		//read in the receiver positions for this shot
		assert(json_src_rec[thisShot].HasMember("z_rec"));
		assert(json_src_rec[thisShot]["z_rec"].IsArray());
		const Value &js_z_rec = json_src_rec[thisShot]["z_rec"];
		for (SizeType ii = 0; ii < js_z_rec.Size(); ii++) {
			h_z_rec[ii] = js_z_rec[ii].GetInt() + nPml;
			// printf("js_z_rec[%d] = %d\n", ii, js_z_rec[ii].GetInt());
		}

		assert(json_src_rec[thisShot].HasMember("x_rec"));
		assert(json_src_rec[thisShot]["x_rec"].IsArray());
		const Value &js_x_rec = json_src_rec[thisShot]["x_rec"];
		for (SizeType ii = 0; ii < js_x_rec.Size(); ii++) {
			h_x_rec[ii] = js_x_rec[ii].GetInt() + nPml;
			// printf("js_x_rec[%d] = %d\n", ii, h_x_rec[ii]);
		}

		// vec_z_rec.push_back(h_z_rec);
		// vec_x_rec.push_back(h_x_rec);

		// get receiver z positions for each shot
		CHECK(cudaMalloc((void **)&d_z_rec, nrec * sizeof(int)));
		CHECK(cudaMemcpy(d_z_rec, h_z_rec, nrec * sizeof(int), cudaMemcpyHostToDevice));
		d_vec_z_rec.push_back(d_z_rec);

		// get receiver x positions for each shot
		CHECK(cudaMalloc((void **)&d_x_rec, nrec * sizeof(int)));
		CHECK(cudaMemcpy(d_x_rec, h_x_rec, nrec * sizeof(int), cudaMemcpyHostToDevice));
		d_vec_x_rec.push_back(d_x_rec);

		// TODO:modify to use input stf
		// get the source time function for each shot
		h_source = new float[nSteps];
		for (int it = 0; it < nSteps; it++){
			// the stf shot index starts from 0!!!
			h_source[it] = stf(shot_ids[i], it);
		}

		CHECK(cudaMalloc((void **)&d_source, nSteps * sizeof(float)));
		CHECK(cudaMemcpy(d_source, h_source, nSteps * sizeof(float), cudaMemcpyHostToDevice));
		cuda_window<<<(nSteps+31)/32,32>>>(nSteps, 1, dt, 0.001, d_source);
		// bp_filter1d(nSteps, dt, 1, d_source, para.filter(), (nSteps+31)/32, 32);
		CHECK(cudaMemcpy(h_source, d_source, nSteps * sizeof(float), cudaMemcpyDeviceToHost));
		vec_source.push_back(h_source);
		d_vec_source.push_back(d_source);
		// displayArray("h_source", h_source, nSteps, 1);


		// assert(json_src_rec[thisShot].HasMember("Src_fname"));
		// assert(json_src_rec[thisShot]["Src_fname"].IsString());
		// cout << "	Src_fname = " << json_src_rec[thisShot]["Src_fname"].GetString() << endl;
		// fileBinLoad(h_source+i, nSteps, json_src_rec[thisShot]["Src_fname"].GetString());




		// get the window for each shot
		if (if_win_) {
			h_win_start = new float[nrec];
			h_win_end = new float[nrec];
			assert(json_src_rec[thisShot].HasMember("win_start"));
			assert(json_src_rec[thisShot]["win_start"].IsArray());
			const Value &js_win_start = json_src_rec[thisShot]["win_start"];
			for (SizeType ii = 0; ii < js_win_start.Size(); ii++) {
				h_win_start[ii] = js_win_start[ii].GetDouble();
				// printf("h_win_start[%d] = %d\n", ii, h_win_start[ii]);
			}
			// 
			assert(json_src_rec[thisShot].HasMember("win_end"));
			assert(json_src_rec[thisShot]["win_end"].IsArray());
			const Value &js_win_end = json_src_rec[thisShot]["win_end"];
			for (SizeType ii = 0; ii < js_win_end.Size(); ii++) {
				h_win_end[ii] = js_win_end[ii].GetDouble();
				// printf("h_win_end[%d] = %d\n", ii, h_win_end[ii]);
			}
			// 
			CHECK(cudaMalloc((void **)&d_win_start, nrec * sizeof(float)));
			CHECK(cudaMemcpy(d_win_start, h_win_start, nrec * sizeof(float), cudaMemcpyHostToDevice));
			d_vec_win_start.push_back(d_win_start);
			// 
			CHECK(cudaMalloc((void **)&d_win_end, nrec * sizeof(float)));
			CHECK(cudaMemcpy(d_win_end, h_win_end, nrec * sizeof(float), cudaMemcpyHostToDevice));
			d_vec_win_end.push_back(d_win_end);

			delete [] h_win_start;
			delete [] h_win_end;
		}
	
		// get weights
		h_weights = new float[nrec];
		if(json_src_rec[thisShot].HasMember("weights")){
			assert(json_src_rec[thisShot]["weights"].IsArray());
			const Value &js_weights = json_src_rec[thisShot]["weights"];
			for (SizeType ii = 0; ii < js_weights.Size(); ii++) {
				h_weights[ii] = js_weights[ii].GetDouble();
				// printf("h_win_start[%d] = %d\n", ii, h_win_start[ii]);
			}
			CHECK(cudaMalloc((void **)&d_weights, nrec * sizeof(float)));
			CHECK(cudaMemcpy(d_weights, h_weights, nrec * sizeof(float), cudaMemcpyHostToDevice));
			d_vec_weights.push_back(d_weights);
			delete [] h_weights;
		} else{
			CHECK(cudaMalloc((void **)&d_weights, nrec * sizeof(float)));
			intialArrayGPU<<<1,512>>>(d_weights, nrec, 1, 1.0);
			d_vec_weights.push_back(d_weights);
		}

		// DL 09/28/2019
		if (json_src_rec[thisShot].HasMember("src_weight")) {
			assert(json_src_rec[thisShot]["src_weight"].IsNumber());
			vec_srcweights.push_back(json_src_rec[thisShot]["src_weight"].GetDouble());
		} else{
			vec_srcweights.push_back(1.0);
		}


		// initialize the host side data cube
		cudaHostAlloc( (void**)&h_data, nSteps*nrec*sizeof(float), cudaHostAllocDefault); //test
		initialArray(h_data, nSteps*nrec, 0.0);
		vec_data.push_back(h_data);

		// // initialize the device side data cube | commented DL 08/09/2019
		// 	CHECK(cudaMalloc((void **)&d_data, nrec * nSteps * sizeof(float)));
		// 	intialArrayGPU<<<blocks,threads>>>(d_data, nrec, nSteps, 0.0);
		// 	d_vec_data.push_back(d_data);

		if (para.if_res()) {
			// initialize the host side observed data cube
			cudaHostAlloc( (void**)&h_data_obs, nSteps*nrec*sizeof(float), cudaHostAllocDefault);
			initialArray(h_data_obs, nSteps*nrec, 0.0);

			vec_data_obs.push_back(h_data_obs);
			
			// // initialize the device side observed data cube | commented DL 08/09/2019
			// CHECK(cudaMalloc((void **)&d_data_obs, nrec * nSteps * sizeof(float)));
			// intialArrayGPU<<<blocks,threads>>>(d_data_obs, nrec, nSteps, 0.0);
			// // CHECK(cudaMemcpyAsync(d_data_obs, h_data_obs, nrec * nSteps * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
			// d_vec_data_obs.push_back(d_data_obs);

			// initialize the host side data residual
			cudaHostAlloc( (void**)&h_res, nSteps*nrec*sizeof(float), cudaHostAllocDefault); //test
			initialArray(h_res, nSteps*nrec, 0.0);
			vec_res.push_back(h_res);
			// // initialize the device side data residual | commented DL 08/09/2019
			// CHECK(cudaMalloc((void **)&d_res, nrec * nSteps * sizeof(float)));
			// intialArrayGPU<<<blocks,threads>>>(d_res, nrec, nSteps, 0.0);
			// d_vec_res.push_back(d_res);

		}
		// load sxx/szz ratio
		if (json_src_rec[thisShot].HasMember("src_rxz")) {
			assert(json_src_rec[thisShot]["src_rxz"].IsNumber());
			vec_src_rxz.push_back(json_src_rec[thisShot]["src_rxz"].GetDouble());
			// printf("src_rxz[%d] = %.*f\n", i, json_src_rec[thisShot]["src_rxz"].GetDouble());
		} else {
			vec_src_rxz.push_back(RSXXZZ);
		}
		h_rec_rxz = new double[nrec];
		if (json_src_rec[thisShot].HasMember("rec_rxz")) {
			const Value &js_rec_rxz = json_src_rec[thisShot]["rec_rxz"];
			for (SizeType ii = 0; ii < js_rec_rxz.Size(); ii++) {
				h_rec_rxz[ii] = js_rec_rxz[ii].GetDouble();
				// printf("h_rec_rxz[%d] = %f\n", ii, h_rec_rxz[ii]);
			}
		} else {
			for (int ii = 0; ii < nrec; ii++) {
				h_rec_rxz[ii] = RSXXZZ;
			}
		}
		vec_rec_rxz.push_back(h_rec_rxz);

	}
#ifdef VERBOSE
	cout << "	vec_source number = " << vec_source.size() << endl;
#endif


}


Src_Rec::~Src_Rec() {

	for(int i=0; i<d_vec_x_rec.size(); i++) {
		CHECK(cudaFree(d_vec_z_rec.at(i)));
		CHECK(cudaFree(d_vec_x_rec.at(i)));
	}

	for(int i=0; i<vec_source.size(); i++) {
		delete [] vec_source.at(i);
		delete [] vec_rec_rxz.at(i);
		CHECK(cudaFree(d_vec_source.at(i)));
	}
	// delete [] vec_source.at(0);

	for(int i=0; i<vec_data.size(); i++) {
		// delete [] vec_data.at(i); //test
		CHECK(cudaFreeHost(vec_data.at(i))); //test
	}

	if (if_res_) {
		for(int i=0; i<vec_data_obs.size(); i++) {
			// delete [] vec_data_obs.at(i); //test
			CHECK(cudaFreeHost(vec_data_obs.at(i)));
			CHECK(cudaFreeHost(vec_res.at(i)));
			// CHECK(cudaFree(d_vec_data_obs.at(i)));
			// CHECK(cudaFree(d_vec_res.at(i)));
		}
	}


	// for(int i=0; i<d_vec_data.size(); i++) {
	// 	CHECK(cudaFree(d_vec_data.at(i)));
	// }


	if (if_win_) {
		for(int i=0; i<d_vec_win_start.size(); i++) {
			CHECK(cudaFree(d_vec_win_start.at(i)));
			CHECK(cudaFree(d_vec_win_end.at(i)));
		}
	}
	for(int i=0; i<d_vec_weights.size(); i++) {
		CHECK(cudaFree(d_vec_weights.at(i)));
	}

	CHECK(cudaFree(d_coef));

}