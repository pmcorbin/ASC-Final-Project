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
    double *filter, *old_R, *old_B, *old_G, *new_R, *new_B, *new_G;
    int i, j;
    Matrix M_filter, M_old_R, M_old_B, M_old_G, M_new_R, M_new_B, M_new_G;

	/*Begin: read in a jpg image*/
    int pixel_size;
    int pixel = 0;
    bundle  RGB_bundle;
	
    // Read jpg to bundle
    read_jpg("Sunflower.jpg", &RGB_bundle);
    pixel_size = RGB_bundle.num_channels;
	
	old_R = (double*)malloc(RGB_bundle.width*RGB_bundle.height*sizeof(double));
    old_G = (double*)malloc(RGB_bundle.width*RGB_bundle.height*sizeof(double));

	// Convert bundle to RGB matices
    for(int i = 0; i < RGB_bundle.width; i++)
    {
        for(int j = 0; j < RGB_bundle.height; j++)
        {
            old_R[i*RGB_bundle.width + j] = RGB_bundle.image_data[pixel*pixel_size];
			old_G[i*RGB_bundle.width + j] = RGB_bundle.image_data[pixel*pixel_size+1];
			old_B[i*RGB_bundle.width + j] = RGB_bundle.image_data[pixel*pixel_size+2];
			pixel++;
        }
    }
	M_old_R.elements = old_R;
	M_old_B.elements = old_B;
	M_old_G.elements = old_G;
	M_old_R.width = RGB_bundle.width;
	M_old_B.width = RGB_bundle.width;
    M_old_G.width = RGB_bundle.width;
	M_old_R.height = RGB_bundle.height;
	M_old_B.height = RGB_bundle.height;
	M_old_G.height = RGB_bundle.height;
	/*End: read in a jpg image*/

    // Filter Function
    int kernelsize = 3;
	M_filter.width = kernelsize;
	M_filter.height = kernelsize;

    for(int i=0;i<M_filter.width;i++){
        for(int j=0;j<M_filter.height;j++){
            filter[i*RGB_bundle.width + j] = 1.0/(kernelsize*kernelsize);
        }
    }
	M_filter.elements = filter;

	struct timeval tvalBefore, tvalAfter;
    gettimeofday (&tvalBefore, NULL);
	////////////////////
	gettimeofday (&tvalAfter, NULL);
    printf("Sequential Time: %f",
            (float)((tvalAfter.tv_sec - tvalBefore.tv_sec))
          );


    MatMul(M_A, M_B,  M_C);

    free(A); free(B); free(C); free(cpu_C); 
    return 0;
}

