#ifndef LIBCUFD_H
#define LIBCUFD_H

#include <string>
using std::string;
extern "C" void cufd(float *misfit, float *grad_Lambda, float *grad_Mu,
                     float *grad_Den, float *grad_stf, const float *Lambda,
                     const float *Mu, const float *Den, const float *stf,
                     int calc_id, const int gpu_id, const int group_size,
                     const int *shot_ids, const string para_fname);
#endif