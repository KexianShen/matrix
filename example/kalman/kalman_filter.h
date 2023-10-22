#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

#include "../../matrix.h"

class KalmanFilter {
 private:
  Matrix<double, 8, 1> state_;                // Current state_ estimate
  Matrix<double, 8, 8> covariance_;           // Error covariance_ matrix
  Matrix<double, 8, 8> transitionMatrix_;     // State transition matrix
  Matrix<double, 6, 8> measurementMatrix_;    // Measurement matrix
  Matrix<double, 6, 6> measurementNoiseCov_;  // Measurement noise covariance
  Matrix<double, 8, 8> processNoiseCov_;      // Process noise covariance

 public:
  KalmanFilter(const Matrix<double, 8, 1>& initialState,
               const Matrix<double, 8, 8>& initialCovariance,
               const Matrix<double, 8, 8>& initialTransition,
               const Matrix<double, 6, 8>& initialMeasurement,
               const Matrix<double, 6, 6>& initialMeasurementNoiseCov,
               const Matrix<double, 8, 8>& initialProcessNoiseCov);

  Matrix<double, 8, 1> getState() const;

  void predict(const Matrix<double, 6, 1>& measurement, const double dt);
};

#endif  // KALMAN_FILTER_HPP