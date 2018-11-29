#if !defined(_POINTER_2D_MATRIX_H)
#define _POINTER_2D_MATRIX_H

#include <iostream>
#include <new>
#include <cstddef>
#include <fstream>
#include "Vector.h"

using namespace std;

#define  off_set            2
#define  Mesh_Size          10
#define  FILTER_SIZE        3

class Matrix{
    private:
    double  **pptr; 
    int     rows, cols; 
    
    public:
    Matrix(int _rows, int _cols)
    {
        rows = _rows;
        cols = _cols;

        pptr = new double*[rows+2*off_set];

        for(int i = 0; i < rows+2*off_set; i++)
            pptr[i] = new double[cols+2*off_set];
    }

    Matrix()
    {
        rows = cols = 0;
        pptr = nullptr;  // use -std=c++11 option in compilation
    }
 	
	// Task 2   
    Matrix(const Matrix&  a)  
	{
    	rows=a.rows;
		cols=a.cols;
		pptr = new double*[rows+2*off_set];
		for(int k = 0; k < rows+2*off_set; k++){
			pptr[k] = new double[cols+2*off_set];
		}
		for(int i=0; i<rows+2*off_set; i++){
			for(int j=0; j<cols+2*off_set; j++){
				pptr[i][j]=a.pptr[i][j];
			}
		}
    	std::cout<<"call Matrix copy constructor"<<std::endl;
	}
	
	// Task 3
	Matrix& operator=(const Matrix&  a)  
	{
    	if(this != &a && NULL != pptr)
       		delete [] pptr;
    	rows=a.rows;
        cols=a.cols;
        pptr = new double*[rows+2*off_set];
        for(int k = 0; k < rows+2*off_set; k++){
            pptr[k] = new double[cols+2*off_set];
        }   
        for(int i=0; i<rows+2*off_set; i++){
            for(int j=0; j<cols+2*off_set; j++){
                pptr[i][j]=a.pptr[i][j];
            }   
        }   
        std::cout<<"call Matrix assignment"<<std::endl;
    	return *this;
	}

    ~Matrix()
    {
        int   i;
        if(nullptr != pptr)
        {
            for(i = 0; i < rows+2*off_set; i++)
                delete[] pptr[i]; 
            delete[] pptr; 
        }
        rows = cols = 0;
    }

    /* Note: operator[] always takes exactly one parameter,
    * but operator() can take any number of parameters
    * (in the case of a rectangular matrix, two parameters are needed). */
    double& operator()(int i, int j)
    {
        return pptr[(i)+off_set][(j)+off_set]; 
    }

    friend ostream& operator<<(std::ostream &output, Matrix &mat)
    {
        int i, j;
        for(i = 0; i < mat.rows; i++)
        {
            for(j = 0; j < mat.cols; j++)
                output<<mat.pptr[(i)+off_set][(j)+off_set]<<" ";
            output<<endl; 
        }
        return output;
    }

    // Cumulative addition of this matrix and another
    // usage: A += B; 
    Matrix& operator+=(const Matrix& rhs)
    {
        int rows = rhs.get_rows();
        int cols = rhs.get_cols();

        for (int i=0; i<rows; i++) {
          for (int j=0; j<cols; j++) {
            pptr[i+off_set][j+off_set] += rhs.pptr[i+off_set][j+off_set];
          }
        }
        return *this;
    }

    // usage C = A + B; 
	// Task 4
	Matrix operator+(const Matrix& a)   // Addition of two matrices
	{
    	int rows, cols; 
		rows = a.rows;
		cols = a.cols;
    	Matrix new_a(rows,cols);
		for (int i=0; i<rows; i++) {
          	for (int j=0; j<cols; j++) {
          		new_a.pptr[i+off_set][j+off_set]=this->pptr[i+off_set][j+off_set]
					+a.pptr[j+off_set][j+off_set];
			}
        }
		return new_a;
	}
	
	
    friend Vector operator*(Matrix&, Vector&);
    unsigned get_rows() const
    {
        return rows;
    }
    unsigned get_cols() const
    {
        return cols;
    }
	
	// Task 5
	void resize(int n_rows,int n_cols){
		double** old_pptr = pptr;
		int old_rows = rows;
		int old_cols = cols;
		rows =n_rows;
		cols =n_cols;
		delete[] pptr;
		pptr = new double*[n_rows+2*off_set];
        for(int k = 0; k < n_rows+2*off_set; k++){
            pptr[k] = new double[n_cols+2*off_set];
        }   
        for(int i=0; i<n_rows+2*off_set; i++){
            for(int j=0; j<n_cols+2*off_set; j++){
                pptr[i][j]=old_pptr[i][j];
            }   
        }   
		cout<<"Resize function called"<<endl;	
	}

}; 
#endif 
