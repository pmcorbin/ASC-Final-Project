Compiling: icpc rw_jpg.cpp  pointer_2d_matrix.cpp Vector.cpp -ljpeg -std=c++11

main() function in pointer_2d_matrix.cpp

To run the program: ./a.out Sunflower.jpg

-------------------------

Task 1. implement the matrix vector multiploication by overloading operator* again.
The prototype of the function is:
Vector operator*(Matrix& A, Vector& v);
This function is a friend to both Matrix and Vector classes.

Using this function, we can do:
=Matrix A(4,4); 
Vector v(4), ans_v;
Vector ans_v  A*v; 

---------------------------------
Task 2. Implement the copy constructor for Matrix. 
-----------------------------------

Task 3. implement "assignment functionality which does cleanup and copy" by 
overloading operator=
The prototype of the function is:
Matrix& Matrix::operator=(const Matrix &rhs);

Suppose + is overloaded. Using this function, we can do:
Matrix A(4,4), B(4,4), C;
C = A + B;

--------------------------------
Task 4. Implement the addition of two matrices by overloading operator+.
The prototype of the function is:
Matrix Matrix:: operator+(const Matrix& rhs);  // Addition of two matrices

--------------------------------
Task 5. Implement  function to resize the matrix. 
The prototype of the function is:
void Matrix::resize(int n_rows,int n_cols);

This function allocate a new chunck of memory for storing matrix of size n_rows by n_cols.
The current matrix entry values should be copied into the new memory space. Then old memory is freed. 

-------------------------
Task 6. Implement a filter function for filtering the jpeg image: Sunflower.jpg

The current code read in this image by calling function 

read_jpg(filename, &RGB_bundle);

On return, the RGB_bundle struct stores the RGB values of the image by array pointed by member pointer image_data of the struct.  
The double for loop shows how to extract RGB values for each pixels. 

You need:
 1. Use matrices to store RGB values respectively. Each matric only saves one color. 
 2. Use the following 3 by 3 filter to blur the image: 
     1/9{{1 1 1}; {1 1 1}; {1 1 1}}. 
    Multiply each element of the kernel with its corresponding element of the image matrix (the one which is overlapped with it)
Sum up all product outputs and put the result in the output matrix.
 3. Write the blurred image to a file by call function write_jpg(const char*,bundle*);

 

