#include "kalman_filter.h"

KalmanFilter::KalmanFilter(
    const Matrix<double, 8, 1>& initialState,
    const Matrix<double, 8, 8>& initialCovariance,
    const Matrix<double, 8, 8>& initialTransition,
    const Matrix<double, 6, 8>& initialMeasurement,
    const Matrix<double, 6, 6>& initialMeasurementNoiseCov,
    const Matrix<double, 8, 8>& initialProcessNoiseCov)
    : state_(initialState),
      covariance_(initialCovariance),
      transitionMatrix_(initialTransition),
      measurementMatrix_(initialMeasurement),
      measurementNoiseCov_(initialMeasurementNoiseCov),
      processNoiseCov_(initialProcessNoiseCov) {}

Matrix<double, 8, 1> KalmanFilter::getState() const { return state_; }

void KalmanFilter::predict(const Matrix<double, 6, 1>& measurement,
                           const double dt) {
  transitionMatrix_(0, 1) = dt;
  transitionMatrix_(1, 2) = dt;
  transitionMatrix_(3, 4) = dt;
  transitionMatrix_(4, 5) = dt;
  transitionMatrix_(6, 7) = dt;
  state_ = transitionMatrix_ * state_;
  covariance_ = transitionMatrix_ * covariance_ * transitionMatrix_.t() +
                processNoiseCov_;
  Matrix<double, 6, 1> innovation = measurement - measurementMatrix_ * state_;
  Matrix<double, 6, 6> innovationCov =
      measurementMatrix_ * covariance_ * measurementMatrix_.t() +
      measurementNoiseCov_;
  Matrix<double, 8, 6> kalmanGain =
      covariance_ * measurementMatrix_.t() * innovationCov.inv();
  state_ = state_ + kalmanGain * innovation;
  covariance_ = covariance_ - kalmanGain * measurementMatrix_ * covariance_;
}

int main() {
  // Define initial matrices for Kalman filter
  Matrix<double, 8, 1> initialState{
      0, 12, 0.1, 0, 0.2, 0.1, 3.14, 0.12,
  };
  Eye<double, 8> initialCovariance;
  Eye<double, 8> initialTransition;
  Matrix<double, 6, 8> initialMeasurement;
  initialMeasurement(0, 1) = 1;
  initialMeasurement(1, 2) = 1;
  initialMeasurement(2, 4) = 1;
  initialMeasurement(3, 5) = 1;
  initialMeasurement(4, 6) = 1;
  initialMeasurement(5, 7) = 1;
  Eye<double, 6> initialMeasurementNoiseCov;
  initialMeasurementNoiseCov *= 0.1;
  Eye<double, 8> initialProcessNoiseCov;
  initialProcessNoiseCov *= 0.01;

  // Initialize the Kalman filter
  KalmanFilter kf(initialState, initialCovariance, initialTransition,
                  initialMeasurement, initialMeasurementNoiseCov,
                  initialProcessNoiseCov);

  // Perform Kalman filter operations
  Matrix<double, 6, 1> measurement{
      12.0, 0.2, 0.1, 0.1, 3.141, 0.11,
  };
  double dt = 0.15;

  kf.predict(measurement, dt);

  std::cout << kf.getState() << std::endl;

  return 0;
}
