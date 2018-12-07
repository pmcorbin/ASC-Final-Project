#ifndef FCNS_H
#define FCNS_H

#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Thread block size
#define BLOCK_SIZE      32  // number of threads in a direction of the block
#define A_HEIGHT        2000 // number of columns. 512
#define AB_SHARED_DIM   2000
#define B_WIDTH         2000 // number of rows


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
