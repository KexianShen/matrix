#include <iostream>
#include <vector>

#include "../../matrix.h"

int main() {
  // Creating Matrices
  // -----------------

  // Create a 2x2 matrix with double values, initialized to 0
  Matrix<double, 2, 2> mat1;

  // Create a 2x2 matrix with doubleeger values using an initializer list
  Matrix<double, 2, 2> mat2 = {1, 2, 3, 4};

  // Basic Matrix Operations
  // ------------------------

  // Matrix addition
  Matrix<double, 2, 2> sum = mat1 + mat2;

  // Matrix subtraction
  Matrix<double, 2, 2> diff = mat2 - mat1;

  // Matrix unary minus
  Matrix<double, 2, 2> negated = -mat2;

  // Matrix-scalar multiplication
  Matrix<double, 2, 2> scaled = mat2 * 2;

  // Matrix-scalar multiplication assignment
  mat2 *= 3;

  // Transpose a matrix
  Matrix<double, 2, 2> transposed = mat2.t();

  // Matrix-Matrix Multiplication
  Matrix<double, 2, 2> mat3 = {1, 2, 3, 4};
  Matrix<double, 2, 2> result3 = mat3 * mat2;

  // Matrix Inversion
  Matrix<double, 2, 2> inverted = mat3.inv();

  // Accessing Matrix Elements
  // -------------------------

  // Access elements using parentheses
  double element = mat2(0, 1);  // Accesses the element at row 0, column 1

  // Access elements using data()
  double* dataPtr = mat2.data();
  double thirdElement = dataPtr[2];

  // Printing a Matrix
  // -----------------

  // You can use the overloaded << operator to print a matrix
  std::cout << "Matrix mat2:\n" << mat2 << std::endl;


  return 0;
}
