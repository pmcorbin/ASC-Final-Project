#ifndef FCNS_H
#define FCNS_H

#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Thread block size
#define BLOCK_SIZE      8  // number of threads in a direction of the block
#define KERNELSIZE		10

// Declaration fo Matrix Class
struct Matrix{
    int width;
    int height;
    double* elements;
};

// Declaration of matrix filter kernal function
__global__ void MatFilterKernel(const Matrix,  const Matrix, Matrix);

// Declaration of matrix filter cpu function
float MatFilter(const Matrix myfilter, Matrix oldimage, Matrix newimage);

// Declaration of matrix print function
void MatPrint(const Matrix M);
#endif
