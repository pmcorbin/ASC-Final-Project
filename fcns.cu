#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "fcns.h"
#include <stdio.h>
#include <stdlib.h>

// Definition of Matrix Filter Function CPU
void MatFilter(const Matrix filter,const Matrix oldmat, Matrix newmat)
{
    // Load  to device memory
    Matrix d_filter;
    d_filter.width = filter.width; 
	d_filter.height = filter.height;
    size_t size = filter.width * filter.height * sizeof(double);
    cudaMalloc(&d_filter.elements, size);
    cudaMemcpy(d_filter.elements, filter.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_oldmat;
    d_oldmat.width = oldmat.width; 
	d_oldmat.height = oldmat.height;
    size = oldmat.width * oldmat.height * sizeof(double);
    cudaMalloc(&d_oldmat.elements, size);
    cudaMemcpy(d_oldmat.elements, oldmat.elements, size,
               cudaMemcpyHostToDevice);
	Matrix d_newmat;
    d_newmat.width = newmat.width;
    d_newmat.height = newmat.height;
    size = newmat.width * newmat.height * sizeof(double);
    cudaMalloc(&d_newmat.elements, size);
	
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((newmat.width+BLOCK_SIZE-1) / dimBlock.x, 
		(newmat.height+BLOCK_SIZE-1)/ dimBlock.y);

    MatFilterKernel<<<dimGrid, dimBlock>>>(d_filter, d_oldmat, d_newmat);

    // Read C from device memory
    cudaMemcpy(newmat.elements, d_newmat.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_filter.elements);
    cudaFree(d_oldmat.elements);
    cudaFree(d_newmat.elements);
}

// Definition of Matrix Filter Function GPU
__global__ void MatFilterKernel(Matrix filter, Matrix oldmat, Matrix newmat)
{
    // Each thread computes one element of newmat
    // by accumulating results into tempval
    double tempval = 0;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Apply Filter by row and column
   	for(int k=0;k<filter.height;k++){
        for(int m=0; m<filter.width; m++){
        	tempval+=oldmat.elements[(row+k)*oldmat.width+(col+m)]
					*filter.elements[k*filter.width+m];
    	}
   	}
	newmat.elements[row * newmat.width + col] = tempval;
}

void MatPrint(const Matrix M){
	for(int i=0; i<M.height; i++){
		for(int j=0; j<M.width; j++){
			printf("%f ",M.elements[i*M.width+j]);
		}	
		printf("\n");
	}
}



