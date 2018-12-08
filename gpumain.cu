#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <new>
#include <cstddef>
#include <fstream>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <sys/time.h>
#include "rw_jpg.h"
#include "fcns.h"

int main(){
	/*///////////////                           /////////////////////
    /////////////////   INITIALIZE VARIABLES  	/////////////////////
    ////////////////                            ///////////////////// */
    double *filter, *old_R, *old_B, *old_G, *new_R, *new_B, *new_G;
	double *gpu_new_R, *gpu_new_G, *gpu_new_B;
    Matrix M_filter, M_old_R, M_old_B, M_old_G, M_new_R, M_new_B, M_new_G;
	Matrix M_gpu_new_R, M_gpu_new_B, M_gpu_new_G;

	/*///////////////                           /////////////////////
    /////////////////   READ IN JPEG  IMAGE  	/////////////////////
    ////////////////                            ///////////////////// */
    int pixel_size;
    int pixel = 0;
    bundle  RGB_bundle;
	
    // Read jpg to bundle
    read_jpg("5000.jpg", &RGB_bundle);
    pixel_size = RGB_bundle.num_channels;
	
	// Allocate memory to store RGB values
	old_R = (double*)malloc(RGB_bundle.width*RGB_bundle.height*sizeof(double));
    old_G = (double*)malloc(RGB_bundle.width*RGB_bundle.height*sizeof(double));
	old_B = (double*)malloc(RGB_bundle.width*RGB_bundle.height*sizeof(double));
		
	// Convert bundle to RGB matices
    for(int i = 0; i < RGB_bundle.height; i++)
    {
        for(int j = 0; j < RGB_bundle.width; j++)
        {
			old_R[i*RGB_bundle.width + j] = RGB_bundle.image_data[pixel*pixel_size];
			old_G[i*RGB_bundle.width + j] = RGB_bundle.image_data[pixel*pixel_size+1];
			old_B[i*RGB_bundle.width + j] = RGB_bundle.image_data[pixel*pixel_size+2];
			pixel++;
        }
    }
	
	// Move RGB elements into matrix struct
	M_old_R.elements = old_R;
	M_old_B.elements = old_B;
	M_old_G.elements = old_G;
	M_old_R.width = RGB_bundle.width;
	M_old_B.width = RGB_bundle.width;
    M_old_G.width = RGB_bundle.width;
	M_old_R.height = RGB_bundle.height;
	M_old_B.height = RGB_bundle.height;
	M_old_G.height = RGB_bundle.height;
    
	/*///////////////                           		/////////////////////
    /////////////////   INITIALIZE FILTER FUNCTION  	/////////////////////
    ////////////////                            		///////////////////// */
    int kernelsize = 10; // Filter height and width
	
	// Allocate memory for filter matrix
	M_filter.width = kernelsize;
	M_filter.height = kernelsize;
	filter = (double*)malloc(kernelsize*kernelsize*sizeof(double));
	
	// Populate filter matrix
	for(int i=0;i<M_filter.height;i++){
        for(int j=0;j<M_filter.width;j++){
            filter[i*kernelsize + j] = 1.0/(kernelsize*kernelsize);
		}
    }
	
	// Move filter values to matrix struct
	M_filter.elements = filter;
	
	/*///////////////                           /////////////////////
    /////////////////   CPU FILTERING OF IMAGE  /////////////////////
    ////////////////                            ///////////////////// */
	
	// Allocate memory for filtered RGB matrices
	new_R = (double*)malloc(RGB_bundle.width*RGB_bundle.height*sizeof(double)-kernelsize+1);
    new_B = (double*)malloc(RGB_bundle.width*RGB_bundle.height*sizeof(double)-kernelsize+1);
    new_G = (double*)malloc(RGB_bundle.width*RGB_bundle.height*sizeof(double)-kernelsize+1);
	M_new_R.width = RGB_bundle.width-kernelsize+1;
    M_new_B.width = RGB_bundle.width-kernelsize+1;
    M_new_G.width = RGB_bundle.width-kernelsize+1;
    M_new_R.height = RGB_bundle.height-kernelsize+1;
    M_new_B.height = RGB_bundle.height-kernelsize+1;
    M_new_G.height = RGB_bundle.height-kernelsize+1;
	M_gpu_new_R.elements 
	= (double*)malloc(RGB_bundle.width*RGB_bundle.height*sizeof(double)-kernelsize+1);
    M_gpu_new_B.elements 
	= (double*)malloc(RGB_bundle.width*RGB_bundle.height*sizeof(double)-kernelsize+1);
    M_gpu_new_G.elements 
	= (double*)malloc(RGB_bundle.width*RGB_bundle.height*sizeof(double)-kernelsize+1);
    M_gpu_new_R.width = RGB_bundle.width-kernelsize+1;
    M_gpu_new_B.width = RGB_bundle.width-kernelsize+1;
    M_gpu_new_G.width = RGB_bundle.width-kernelsize+1;
    M_gpu_new_R.height = RGB_bundle.height-kernelsize+1;
    M_gpu_new_B.height = RGB_bundle.height-kernelsize+1;
    M_gpu_new_G.height = RGB_bundle.height-kernelsize+1;
	
	// CPU filtering loop
	printf("Start\n");
	struct timeval tvalBefore, tvalAfter;	// for measuring cpu execution time
    gettimeofday (&tvalBefore, NULL);
	float CPUtime=0;
	for(int i=0;i<M_new_R.height;i++){
		for(int j=0;j<M_new_R.width;j++){
			new_R[i*M_new_R.width + j]=0;
			new_B[i*M_new_R.width + j]=0;
			new_G[i*M_new_R.width + j]=0;
			
			// Apply Filter by row and column
			for(int k=0;k<M_filter.height;k++){
				for(int m=0; m<M_filter.width; m++){
					new_R[i*M_new_R.width + j]=new_R[i*M_new_R.width+j]
						+old_R[(i+k)*M_old_R.width+(j+m)]*filter[k*M_filter.width+m];
					new_G[i*M_new_G.width + j]=new_G[i*M_new_G.width+j]
                        +old_G[(i+k)*M_old_G.width+(j+m)]*filter[k*M_filter.width+m];
					new_B[i*M_new_B.width + j]=new_B[i*M_new_B.width+j]
                        +old_B[(i+k)*M_old_B.width+(j+m)]*filter[k*M_filter.width+m];
				}
			}
		}   
	}
	gettimeofday (&tvalAfter, NULL);	// for measuring cpu execution time
	CPUtime = (tvalAfter.tv_sec - tvalBefore.tv_sec) * 1000.0;      // sec to ms
    CPUtime += (tvalAfter.tv_usec - tvalBefore.tv_usec) / 1000.0;
	printf("CPU Time: %f s\n",CPUtime/1000);

	/*///////////////							/////////////////////
	/////////////////	GPU FILTERING OF IMAGE  /////////////////////
	////////////////							///////////////////// */
	float elapseTime1 =MatFilter(M_filter,M_old_R, M_gpu_new_R);
	float elapseTime2 =MatFilter(M_filter, M_old_G, M_gpu_new_G);
	float elapseTime3 =MatFilter(M_filter, M_old_B, M_gpu_new_B);
	printf("GPU Time: %f s\n",elapseTime1+elapseTime2+elapseTime3);
	/*///////////////                           		/////////////////////
    /////////////////   EXPORT FILTERED IMAGE TO JPEG  	/////////////////////
    ////////////////                            		///////////////////// */
	
	// Initialize new bundles for exporting filtered image
	bundle n_RGB_bundle;
	n_RGB_bundle.height = M_gpu_new_R.height;
	n_RGB_bundle.width = M_gpu_new_R.width;
	n_RGB_bundle.num_channels = 3;
	pixel =0;
	n_RGB_bundle.image_data = (unsigned char*) malloc(n_RGB_bundle.width*n_RGB_bundle.height*pixel_size);
	
	// Move matrix vaules to bundle
	for(int i = 0; i < n_RGB_bundle.height; i++)
    {
        for(int j = 0; j < n_RGB_bundle.width; j++)
        {
            n_RGB_bundle.image_data[pixel*pixel_size]=M_gpu_new_R.elements[i*n_RGB_bundle.width + j];
            n_RGB_bundle.image_data[pixel*pixel_size +1]=M_gpu_new_G.elements[i*n_RGB_bundle.width + j];
            n_RGB_bundle.image_data[pixel*pixel_size +2]=M_gpu_new_B.elements[i*n_RGB_bundle.width + j];
            pixel++;
        }
    }
	
	// Write bundle to jpeg image
    write_jpg("5000_2.jpg", &n_RGB_bundle);
    
	/*///////////////                           /////////////////////
    /////////////////   	CLEAN UP		  	/////////////////////
    ////////////////                            ///////////////////// */
	free(RGB_bundle.image_data); 
	free(n_RGB_bundle.image_data);
    free(old_R); free(old_G); free(old_B); 
	free(new_R); free(new_G); free(new_B); 
	//free(gpu_new_R); free(gpu_new_G); free(gpu_new_B);
    return 0;
}

