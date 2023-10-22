#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <stdexcept>
#include <vector>

#ifdef BLAS
#include <cblas.h>
#include <lapacke.h>
#endif

template <typename T, size_t Rows, size_t Cols>
class Matrix {
 protected:
  std::vector<T> data_;

 public:
  Matrix() : data_(Rows * Cols, T()) {
    static_assert(std::is_arithmetic<T>::value, "T must be a numeric type");
  }

  // Copy constructor
  Matrix(const Matrix<T, Rows, Cols>& other) : data_(other.data_) {}

  // Assign constructor
  Matrix& operator=(const Matrix<T, Rows, Cols>& other) {
    if (this != &other) {
      data_ = other.data_;
    }
    return *this;
  }

  // Constructor with initializer list
  Matrix(std::initializer_list<T> init) {
    if (init.size() != Rows * Cols) {
      throw std::invalid_argument("Invalid initializer list size");
    }
    data_.resize(Rows * Cols);
    size_t index = 0;
    for (const auto& value : init) {
      data_[index] = value;
      index++;
    }
  }

  // Constructor with initializer list of initializer list
  Matrix(std::initializer_list<std::initializer_list<T>> init) {
    if (init.size() != Rows || init.begin()->size() != Cols) {
      throw std::invalid_argument("Invalid initializer list size");
    }
    data_.resize(Rows * Cols);
    size_t row = 0;
    for (const auto& row_list : init) {
      if (row_list.size() != Cols) {
        throw std::invalid_argument("Invalid initializer list size");
      }
      size_t col = 0;
      for (const auto& value : row_list) {
        data_[row * Cols + col] = value;
        col++;
      }
      row++;
    }
  }

  // Constructor with vector
  Matrix(const std::vector<T>& dataArray) {
    if (dataArray.size() != Rows * Cols) {
      throw std::runtime_error("Data size doesn't match matrix dimensions.");
    }
    data_.resize(Rows * Cols);
    for (size_t i = 0; i < Rows * Cols; ++i) {
      data_[i] = dataArray[i];
    }
  }

  // Accessor using parentheses
  T& operator()(size_t row, size_t col) {
    if (row > Rows - 1 || col > Cols - 1) {
      throw std::invalid_argument("Index exceeds limits.");
    }
    return data_[row * Cols + col];
  }

  // Accessor using parentheses for const objects
  const T& operator()(size_t row, size_t col) const {
    if (row > Rows - 1 || col > Cols - 1) {
      throw std::invalid_argument("Index exceeds limits.");
    }
    return data_[row * Cols + col];
  }

  // Accessor using data()
  T* data() { return data_.data(); }

  // Accessor using data() for const objects
  const T* data() const { return data_.data(); }

  // Destructor
  ~Matrix() {}

  // Matrix transposition
  Matrix<T, Cols, Rows> t() const {
    Matrix<T, Cols, Rows> result;
    for (size_t i = 0; i < Rows; ++i) {
      for (size_t j = 0; j < Cols; ++j) {
        result.data()[j * Rows + i] = data_[i * Cols + j];
      }
    }
    return result;
  }

  // Matrix << operator
  friend std::ostream& operator<<(std::ostream& os,
                                  const Matrix<T, Rows, Cols>& matrix) {
    for (size_t i = 0; i < Rows; ++i) {
      for (size_t j = 0; j < Cols; ++j) {
        os << matrix.data_[i * Cols + j] << "\t";
      }
      os << "\n";
    }
    return os;
  }

#ifndef BLAS
  // Matrix addition
  Matrix<T, Rows, Cols> operator+(const Matrix<T, Rows, Cols>& other) const {
    Matrix<T, Rows, Cols> result;
    for (size_t i = 0; i < Rows * Cols; ++i) {
      result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
  }

  // Matrix addition assignment
  void operator+=(const Matrix<T, Rows, Cols>& other) {
    for (size_t i = 0; i < Rows * Cols; ++i) {
      data_[i] += other.data_[i];
    }
  }

  // Matrix subtraction
  Matrix<T, Rows, Cols> operator-(const Matrix<T, Rows, Cols>& other) const {
    Matrix<T, Rows, Cols> result;
    for (size_t i = 0; i < Rows * Cols; ++i) {
      result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
  }

  // Matrix unary minus
  Matrix<T, Rows, Cols> operator-() const {
    Matrix<T, Rows, Cols> result;
    for (size_t i = 0; i < Rows * Cols; ++i) {
      result.data_[i] = -data_[i];
    }
    return result;
  }

  // Matrix-Scalar multiplication
  Matrix<T, Rows, Cols> operator*(const T& scalar) const {
    Matrix<T, Rows, Cols> result;
    for (size_t i = 0; i < Rows * Cols; ++i) {
      result.data_[i] = data_[i] * scalar;
    }
    return result;
  }

  // Matrix-Scalar multiplication assignment
  void operator*=(const T& scalar) {
    for (size_t i = 0; i < Rows * Cols; ++i) {
      data_[i] *= scalar;
    }
  }

  // Matrix-Matrix multiplication
  template <size_t OtherCols>
  Matrix<T, Rows, OtherCols> operator*(
      const Matrix<T, Cols, OtherCols>& other) const {
    Matrix<T, Rows, OtherCols> result;
    for (size_t i = 0; i < Rows; ++i) {
      for (size_t j = 0; j < OtherCols; ++j) {
        T sum = 0;
        for (size_t k = 0; k < Cols; ++k) {
          sum += data_[i * Cols + k] * other(k, j);
        }
        result(i, j) = sum;
      }
    }
    return result;
  }

  // Matrix inversion
  Matrix<T, Rows, Cols> inv() const;

#else
  // Matrix addition
  Matrix<T, Rows, Cols> operator+(const Matrix<T, Rows, Cols>& other) const {
    Matrix result(other);
    if constexpr (std::is_same<T, double>::value) {
      cblas_daxpy(Rows * Cols, 1.0, data(), 1, result.data(), 1);
    } else {
      cblas_saxpy(Rows * Cols, 1.0, data(), 1, result.data(), 1);
    }
    return result;
  }

  // Matrix addition assignment
  void operator+=(const Matrix<T, Rows, Cols>& other) {
    if constexpr (std::is_same<T, double>::value) {
      cblas_daxpy(Rows * Cols, 1.0, other.data(), 1, data(), 1);
    } else {
      cblas_saxpy(Rows * Cols, 1.0, other.data(), 1, data(), 1);
    }
  }

  // Matrix subtraction
  Matrix<T, Rows, Cols> operator-(const Matrix<T, Rows, Cols>& other) const {
    Matrix result(*this);
    if constexpr (std::is_same<T, double>::value) {
      cblas_daxpy(Rows * Cols, -1.0, other.data(), 1, result.data(), 1);
    } else {
      cblas_saxpy(Rows * Cols, -1.0, other.data(), 1, result.data(), 1);
    }
    return result;
  }

  // Matrix unary minus
  Matrix operator-() const {
    Matrix result;
    if constexpr (std::is_same<T, double>::value) {
      cblas_daxpy(Rows * Cols, -1.0, data(), 1, result.data(), 1);
    } else {
      cblas_saxpy(Rows * Cols, -1.0, data(), 1, result.data(), 1);
    }
    return result;
  }

  // Matrix-Scalar multiplication
  Matrix<T, Rows, Cols> operator*(const T& scalar) {
    Matrix<T, Rows, Cols> result(*this);
    if constexpr (std::is_same<T, double>::value) {
      cblas_dscal(Rows * Cols, scalar, result.data(), 1);
    } else {
      cblas_sscal(Rows * Cols, scalar, result.data(), 1);
    }
    return result;
  }

  void operator*=(const T& scalar) {
    if constexpr (std::is_same<T, double>::value) {
      cblas_dscal(Rows * Cols, scalar, data(), 1);
    } else {
      cblas_sscal(Rows * Cols, scalar, data(), 1);
    }
  }

  // Matrix-Vector multiplication
  Matrix<T, Rows, 1> operator*(const Matrix<T, Cols, 1>& other) const {
    Matrix<T, Rows, 1> result;
    if constexpr (std::is_same<T, double>::value) {
      cblas_dgemv(CblasRowMajor, CblasNoTrans, Rows, Cols, 1.0f, data(), Cols,
                  other.data(), 1, 0.0f, result.data(), 1);
    } else {
      cblas_sgemv(CblasRowMajor, CblasNoTrans, Rows, Cols, 1.0f, data(), Cols,
                  other.data(), 1, 0.0f, result.data(), 1);
    }
    return result;
  }

  // Matrix-Matrix multiplication
  template <size_t OtherCols>
  Matrix<T, Rows, OtherCols> operator*(
      const Matrix<T, Cols, OtherCols>& other) const {
    Matrix<T, Rows, OtherCols> result;
    if constexpr (std::is_same<T, double>::value) {
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Rows, OtherCols,
                  Cols, 1.0, data(), Cols, other.data(), OtherCols, 0.0,
                  result.data(), OtherCols);
    } else {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Rows, OtherCols,
                  Cols, 1.0, data(), Cols, other.data(), OtherCols, 0.0,
                  result.data(), OtherCols);
    }
    return result;
  }

  // Matrix inversion
  Matrix<T, Rows, Cols> inv() const {
    static_assert(Rows == Cols, "Matrix must be square to calculate inverse.");
    Matrix<T, Rows, Cols> result(*this);
    // Get the matrix dimensions
    lapack_int n = static_cast<lapack_int>(Rows);
    lapack_int lda = n;
    lapack_int info;
    // Perform LU factorization
    std::vector<lapack_int> ipiv(n);
    if constexpr (std::is_same<T, double>::value) {
      info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, result.data(), lda,
                            ipiv.data());
    } else {
      info = LAPACKE_sgetrf(LAPACK_ROW_MAJOR, n, n, result.data(), lda,
                            ipiv.data());
    }
    if (info != 0) {
      throw std::runtime_error("Matrix is singular, cannot be inverted.");
    }
    // Calculate inverse using the LU factors
    if constexpr (std::is_same<T, double>::value) {
      info =
          LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, result.data(), lda, ipiv.data());
    } else {
      info =
          LAPACKE_sgetri(LAPACK_ROW_MAJOR, n, result.data(), lda, ipiv.data());
    }
    if (info != 0) {
      throw std::runtime_error("Error during matrix inversion.");
    }
    return result;
  }
#endif
};

template <typename T, size_t N>
class Eye : public Matrix<T, N, N> {
 public:
  Eye() {
    for (size_t i = 0; i < N; ++i) {
      this->data_[i * N + i] = 1;
    }
  }
};

#ifndef BLAS
template <typename T, size_t Rows, size_t Cols>
Matrix<T, Rows, Cols> Matrix<T, Rows, Cols>::inv() const {
  static_assert(Rows == Cols, "Matrix must be square for inversion");
  // Create an identity matrix
  Eye<T, Rows> identity;
  Matrix<T, Rows, Cols> temp(*this);
  Matrix<T, Rows, Cols> result(identity);
  // Perform Gauss-Jordan elimination with partial pivoting
  for (size_t i = 0; i < Rows; ++i) {
    // Find the pivot row and swap rows
    size_t pivotRow = i;
    for (size_t j = i + 1; j < Rows; ++j) {
      if (std::abs(temp(j, i)) > std::abs(temp(pivotRow, i))) {
        pivotRow = j;
      }
    }
    if (temp(pivotRow, i) == 0) {
      throw std::runtime_error("Matrix is singular, cannot be inverted.");
    }
    if (pivotRow != i) {
      for (size_t j = 0; j < Cols; ++j) {
        std::swap(temp(i, j), temp(pivotRow, j));
        std::swap(result(i, j), result(pivotRow, j));
      }
    }
    // Normalize the pivot row
    T pivot = temp(i, i);
    for (size_t j = 0; j < Cols; ++j) {
      temp(i, j) /= pivot;
      result(i, j) /= pivot;
    }
    // Eliminate other rows
    for (size_t j = 0; j < Rows; ++j) {
      if (j != i) {
        T factor = temp(j, i);
        for (size_t k = 0; k < Cols; ++k) {
          temp(j, k) -= factor * temp(i, k);
          result(j, k) -= factor * result(i, k);
        }
      }
    }
  }
  return result;
}
#endif

#endif  // MATRIX_H
