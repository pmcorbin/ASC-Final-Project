#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <fcns.h>

// Definition of Matrix Type
typedef struct {
    int width;
    int height;
    double* elements;
} Matrix;

// Definition of Matrix Filter Function CPU
void MatFilter(const Matrix myfilter, Matrix oldimage, Matrix newimage)
{
    // Load  to device memory
    Matrix d_myfilter;
    d_myfilter.width = myfilter.width; 
	d_myfilter.height = myfilter.height;
    size_t size = myfilter.width * myfilter.height * sizeof(double);
    cudaMalloc(&d_myfilter.elements, size);
    cudaMemcpy(d_myfilter.elements, myfilter.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_myimage;
    d_oldimage.width = oldimage.width; 
	d_oldimage.height = oldimage.height;
    size = oldimage.width * oldimage.height * sizeof(double);
    cudaMalloc(&d_oldimage.elements, size);
    cudaMemcpy(d_oldimage.elements, oldimage.elements, size,
               cudaMemcpyHostToDevice);
	Matrix d_myimage;
    d_newimage.width = newimage.width;
    d_newimage.height = newimage.height;
    size = newimage.width * newimage.height * sizeof(double);
    cudaMalloc(&d_newimage.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((newimage.width+BLOCK_SIZE-1) / dimBlock.x, 
		(newimage.height+BLOCK_SIZE-1)/ dimBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_filter, d_oldimage, d_newimage);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapseTime;
    cudaEventElapsedTime(&elapseTime, start, stop);
    printf("Time to generate: %f ms\n", elapseTime);

    // Read C from device memory
    cudaMemcpy(newimage.elements, d_newimage.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_filter.elements);
    cudaFree(d_oldimage.elements);
    cudaFree(d_newimage.elements);
}

// Definition of Matrix Filter Function GPU
__global__ void MatFilterKernel(Matrix filter, Matrix oldimage, Matrix newimage)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    double Cvalue = 0;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < A_HEIGHT && col < B_WIDTH)
    {
        for (int i = 0; i < A.width; ++i)
            Cvalue += A.elements[row * A.width + i]
                    * B.elements[i * B.width + col];
        C.elements[row * C.width + col] = Cvalue;
    }
}

