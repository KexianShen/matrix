# Matrix

Matrix is a single-header C++ matrix library that offers a clean interface for working with matrices in your applications. Notably, this library comes with LAPACK support to provide efficient linear algebra operations, all within a single header file for your convenience.

## Usage

### Including the Header

To use the Matrix class in your C++ project, simply include the single header file:

```cpp
#include "matrix.h"
```

If you want to enable BLAS and LAPACK support, you can use the following command after installing [LAPACK](https://github.com/Reference-LAPACK/lapack):

```bash
make blas=1
```

### Creating Matrices

Creating a matrix is straightforward:

```cpp
Matrix<int, 2, 3> matrix;  // Creates a 2x3 matrix of integers, initialized to 0.
```

You can initialize matrices with data using initializer lists:

```cpp
Matrix<double, 2, 2> matrix = {1.0, 2.0, 3.0, 4.0};
```

### Basic Operations

Perform basic operations like addition, subtraction, unary minus, multiplication and inversion:

```cpp
Matrix<double, 2, 2> sum = matrix1 + matrix2;       // Matrix addition
Matrix<double, 2, 2> diff = matrix2 - matrix1;      // Matrix subtraction
Matrix<double, 2, 2> negated = -matrix1;            // Unary minus
Matrix<double, 2, 2> scaled = matrix1 * 2;          // Scalar multiplication
Matrix<double, 2, 2> product = matrix1 * matrix2;   // Matrix multiplication
Matrix<double, 2, 2> inverse = matrix1.inv();       // Matrix inversion
Matrix<double, 2, 2> transposed = matrix1.t()       // Matrix Transposition
```

### Accessing Elements

Access matrix elements using parentheses or `data()` for const objects:

```cpp
double element = matrix(0, 1);  // Accesses the element at row 0, column 1
const double* dataPtr = matrix.data();
double thirdElement = dataPtr[2];  // Accesses the third element
```

### Printing Matrices

Print matrices neatly using the overloaded `<<` operator:

```cpp
std::cout << "Matrix:\n" << matrix << std::endl;
```
