#ifndef FCNS_H
#define FCNS_H

#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Declaration of matrix filter kernal function
__global__ void MatFilterKernel(const Matrix,  const Matrix, Matrix);

// Declaration of matrix filter cpu function
void MatFilter(const Matrix myfilter, Matrix oldimage, Matrix newimage);

#endif
