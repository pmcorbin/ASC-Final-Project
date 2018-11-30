#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <sys/time.h>
#include "rw_jpg.h"

// Thread block size
#define BLOCK_SIZE  	16  // number of threads in a direction of the block
#define A_HEIGHT     	2000 // number of columns. 512
#define AB_SHARED_DIM	2000
#define B_WIDTH    		2000 // number of rows

int main(){
    double *A, *B, *C, *cpu_C;
    int i, j;
    Matrix M_A, M_B, M_C; 
	
    cpu_C = (double*)malloc(A_HEIGHT*B_WIDTH*sizeof(double));  
    C = (double*)malloc(A_HEIGHT*B_WIDTH*sizeof(double));  
    A = (double*)malloc(A_HEIGHT*AB_SHARED_DIM*sizeof(double));
    B = (double*)malloc(AB_SHARED_DIM*B_WIDTH*sizeof(double));

	// initialize cpu_C[]
    for(i = 0; i < A_HEIGHT; i++)
    {
        for(j = 0; j < B_WIDTH; j++)
        {
            cpu_C[i*B_WIDTH + j] = 0.0;
        }
    }


    M_C.width = B_WIDTH; M_C.height = A_HEIGHT;
    M_C.elements = C; 
	
	struct timeval tvalBefore, tvalAfter;
    gettimeofday (&tvalBefore, NULL);
    double Cvalue; 
    for(i = 0; i < A_HEIGHT; i++)
    {
        for(j = 0; j < B_WIDTH; j++)
        {
            Cvalue = 0.0; 
            for (int k = 0; k < AB_SHARED_DIM; ++k)
                Cvalue += M_A.elements[i * M_A.width + k]
                          *M_B.elements[k * M_B.width + j];
            cpu_C[i*B_WIDTH + j] = Cvalue;
        }
    }
	gettimeofday (&tvalAfter, NULL);
    printf("Sequential Time: %f",
            (float)((tvalAfter.tv_sec - tvalBefore.tv_sec))
          );


    MatMul(M_A, M_B,  M_C);

    free(A); free(B); free(C); free(cpu_C); 
    return 0;
}

