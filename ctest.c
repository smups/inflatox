#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

double calc_h10(const double coords[2], const double params[3]) {
  const double φ = coords[0];
  const double ψ = coords[1];

  const double m = params[0];
  const double φ0 = params[1];
  const double ψ0 = params[2];

  return 2.0*ψ0*(φ - φ0)*(2.0*pow(m, 4.0)*pow(φ, 2.0)*(φ - φ0)*(φ*(pow(φ, 2.0) + 1.0) + 2.0*φ - 2.0*φ0 - (φ - φ0)*(pow(φ, 2.0) + 1.0)) - pow(ψ0, 2.0)*pow(pow(φ, 2.0) + 1.0, 2.0))/(pow(φ, 2.0)*(4.0*pow(m, 4.0)*pow(φ, 2.0)*pow(φ - φ0, 2.0) + pow(ψ0, 2.0)*pow(pow(φ, 2.0) + 1.0, 2.0))*(pow(m, 2.0)*pow(φ - φ0, 2.0) - 2.0*ψ*ψ0)*fabs(φ - φ0));
}

int main() {
  printf("hello from c!\n");
  printf("Using %lu bit floats.\n", sizeof(double)*8);
  const double params[3] = { 10.0 /*m*/, 2.0 /*φ0*/, 1.0 /*ψ0*/ };
  const double step = 0.1;
  const double range_start = -10.0;
  const double range_end = -range_start;

  double accumulator = 0.0;

  //Begin timing
  clock_t begin = clock();

  for (double x = range_start; x <= range_end; x += step) {
    for (double y = range_start; y <= range_end; y += step) {
      const double coords[2] = {x, y};
      double result = calc_h10(coords, params);
      //printf("%f ", result);
      accumulator += result;
    }
  }

  clock_t end = clock();
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Elapsed: %f\n", time_spent);
  printf("Answer: %f\n", accumulator);
}
