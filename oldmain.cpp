#include <iostream>
#include <new>
#include <cstddef>
#include <fstream>
#include "rw_jpg.h"
#include "pointer_2d_matrix.h"

using namespace std;

/*Compilation:  icpc read_jpg.cpp  pointer_2d_matrix.cpp -ljpeg -std=c++11 */
int main(int argc, char *argv[]){	
    
	/*Begin: read in a jpg image*/
    int pixel_size;
    int red, green, blue;
    int pixel = 0;
    const char *filename = argv[1];
    bundle  RGB_bundle;

	// Read jpg to bundle
    read_jpg(filename, &RGB_bundle);
    pixel_size = RGB_bundle.num_channels;
	
	// Initialize RGB matices
	Matrix R(RGB_bundle.height,RGB_bundle.width),
			G(RGB_bundle.height,RGB_bundle.width),
			B(RGB_bundle.height,RGB_bundle.width);
	
	// Convert bundle to RGB matices
    for(int i = 0; i < RGB_bundle.height; i++)
    {
        for(int j = 0; j < RGB_bundle.width; j++)
        {
            R(i,j)= RGB_bundle.image_data[pixel*pixel_size];
            G(i,j) = RGB_bundle.image_data[pixel*pixel_size +1];
            B(i,j) = RGB_bundle.image_data[pixel*pixel_size +2];
            pixel++;
        }
    }
    /*End: read in a jpg image*/
	
	// Filter Function
	int kernelsize = 3;
	Matrix mykernel(kernelsize,kernelsize);
	
	for(int i=0;i<mykernel.get_rows();i++){
		for(int j=0;j<mykernel.get_rows();j++){
			mykernel(i,j) = 1.0/(mykernel.get_rows()*mykernel.get_rows());
		}	
	}
	//cout<<"Kernel="<<mykernel<<endl;
	
	// Initialize filtered RGB matrices
	Matrix n_R(RGB_bundle.height-kernelsize+1,RGB_bundle.width-kernelsize+1),
            n_G(RGB_bundle.height-kernelsize+1,RGB_bundle.width-kernelsize+1),
            n_B(RGB_bundle.height-kernelsize+1,RGB_bundle.width-kernelsize+1);
	
	// Filter RGB matrices
	for(int i=0;i<n_R.get_rows();i++){
		for(int j=0;j<n_R.get_cols();j++){
			n_R(i,j)=0;
			// Apply Filter by row and column
			for(int k=0;k<mykernel.get_rows();k++){
				for(int m=0; m<mykernel.get_cols(); m++){
					n_R(i,j) = n_R(i,j)+R(i+k,j+m)*mykernel(k,m);
					n_G(i,j) = n_G(i,j)+G(i+k,j+m)*mykernel(k,m);
					n_B(i,j) = n_B(i,j)+B(i+k,j+m)*mykernel(k,m);
				}
			}
		}   
	}	
	
	// Put Matrices back in bundle
	bundle n_RGB_bundle;
	n_RGB_bundle.height = n_R.get_rows();
	n_RGB_bundle.width = n_R.get_cols();
	n_RGB_bundle.num_channels = 3;
	pixel =0;
	n_RGB_bundle.image_data = new unsigned char [n_RGB_bundle.width*n_RGB_bundle.height*pixel_size];
	for(int i = 0; i < n_RGB_bundle.height; i++)
    {
        for(int j = 0; j < n_RGB_bundle.width; j++)
        {
            n_RGB_bundle.image_data[pixel*pixel_size]=n_R(i,j);
            n_RGB_bundle.image_data[pixel*pixel_size +1]=n_G(i,j);
            n_RGB_bundle.image_data[pixel*pixel_size +2]=n_B(i,j);
            pixel++;
        }
    }
	
			
    /*save jpeg file*/
    write_jpg("Sunflower_2.jpg", &n_RGB_bundle);
    /*END: save jpeg file*/
    
	free(RGB_bundle.image_data); 
	free(n_RGB_bundle.image_data);



