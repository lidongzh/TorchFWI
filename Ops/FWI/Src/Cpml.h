// Dongzhuo Li 05/09/2018
#ifndef CPML_H__
#define CPML_H__

#include "Parameter.h"
#include "Model.h"


class Cpml {

 public:

	Cpml(Parameter &para, Model &model);
	Cpml(const Cpml&) = delete;
	Cpml& operator=(const Cpml&) = delete;

	~Cpml();

	float *K_z, *a_z, *b_z, *K_z_half, *a_z_half, *b_z_half;
	float *K_x, *a_x, *b_x, *K_x_half, *a_x_half, *b_x_half;
	float *d_K_z, *d_a_z, *d_b_z, *d_K_z_half, *d_a_z_half, *d_b_z_half;
	float *d_K_x, *d_a_x, *d_b_x, *d_K_x_half, *d_a_x_half, *d_b_x_half;


};







#endif