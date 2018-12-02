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
    double *filter, *old_R, *old_B, *old_G, *new_R, *new_B, *new_G;
    Matrix M_filter, M_old_R, M_old_B, M_old_G, M_new_R, M_new_B, M_new_G;

	/*Begin: read in a jpg image*/
    int pixel_size;
    int pixel = 0;
    bundle  RGB_bundle;
	
    // Read jpg to bundle
    read_jpg("Sunflower.jpg", &RGB_bundle);
    pixel_size = RGB_bundle.num_channels;
	
	old_R = (double*)malloc(RGB_bundle.width*RGB_bundle.height*sizeof(double));
    old_B = (double*)malloc(RGB_bundle.width*RGB_bundle.height*sizeof(double));
	old_G = (double*)malloc(RGB_bundle.width*RGB_bundle.height*sizeof(double));
		
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
    int kernelsize = 20;
	M_filter.width = kernelsize;
	M_filter.height = kernelsize;
	filter = (double*)malloc(kernelsize*kernelsize*sizeof(double));
	for(int i=0;i<M_filter.height;i++){
        for(int j=0;j<M_filter.width;j++){
            filter[i*kernelsize + j] = 1.0/(kernelsize*kernelsize);
		}
    }
	M_filter.elements = filter;
	//MatPrint(M_filter);
	
	// Filter Image CPU
	new_R = (double*)malloc(RGB_bundle.width*RGB_bundle.height*sizeof(double)-kernelsize+1);
    new_B = (double*)malloc(RGB_bundle.width*RGB_bundle.height*sizeof(double)-kernelsize+1);
    new_G = (double*)malloc(RGB_bundle.width*RGB_bundle.height*sizeof(double)-kernelsize+1);
	M_new_R.width = RGB_bundle.width-kernelsize+1;
    M_new_B.width = RGB_bundle.width-kernelsize+1;
    M_new_G.width = RGB_bundle.width-kernelsize+1;
    M_new_R.height = RGB_bundle.height-kernelsize+1;
    M_new_B.height = RGB_bundle.height-kernelsize+1;
    M_new_G.height = RGB_bundle.height-kernelsize+1;
	
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
	
	// Put Matrices back in bundle
	bundle n_RGB_bundle;
	n_RGB_bundle.height = M_new_R.height;
	n_RGB_bundle.width = M_new_R.width;
	n_RGB_bundle.num_channels = 3;
	pixel =0;
	n_RGB_bundle.image_data = (unsigned char*) malloc(n_RGB_bundle.width*n_RGB_bundle.height*pixel_size);
	for(int i = 0; i < n_RGB_bundle.height; i++)
    {
        for(int j = 0; j < n_RGB_bundle.width; j++)
        {
            n_RGB_bundle.image_data[pixel*pixel_size]=new_R[i*n_RGB_bundle.width + j];
            n_RGB_bundle.image_data[pixel*pixel_size +1]=new_G[i*n_RGB_bundle.width + j];
            n_RGB_bundle.image_data[pixel*pixel_size +2]=new_B[i*n_RGB_bundle.width + j];
            pixel++;
        }
    }
	
	/*save jpeg file*/
    write_jpg("Sunflower_2.jpg", &n_RGB_bundle);
    /*END: save jpeg file*/
    
	free(RGB_bundle.image_data); 
	free(n_RGB_bundle.image_data);
    free(old_R); free(old_G); free(old_B); 
	free(new_R); free(new_G); free(new_B); 
    return 0;
}

