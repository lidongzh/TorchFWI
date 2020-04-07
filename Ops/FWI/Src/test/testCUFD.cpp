#include <iostream>
#include <cmath>
using namespace std;
void cufd(float *res, float *d_Cp,
          const float *Cp, const float *Cs, const float *Den, string dir, int calc_id);

int main()
{
  float res = -1.0;
  int nx = 384, nz = 134;
  float *d_Cp = new float[nx * nz];
  float Cp[nx * nz], Cs[nx * nz], Den[nx * nz];
  for (int i = 0; i < nx * nz; i++)
  {
    Cp[i] = 1000.0;
    Cs[i] = 0.0;
    Den[i] = 1000.0;
  }
  cufd(&res, d_Cp, Cp, Cs, Den, "../params/", 0);
  printf("residual = %f\n", res);

  cufd(&res, d_Cp, Cp, Cs, Den, "../params/", 1);
  double p;
  for (int i = 0; i < nx * nz; i++)
  {
    p += fabs(d_Cp[i]);
  }
  printf("%f\n", p);
  return 1;
}