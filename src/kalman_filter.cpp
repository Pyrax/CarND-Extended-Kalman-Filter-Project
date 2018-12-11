#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Predict() {
  x_ = F_ * x_;  // u is 0
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  * update the state by using Kalman Filter equations
  */
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;

  UpdateByResidual(z, y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  * update the state by using Extended Kalman Filter equations
  * assuming H_ is set to the Jacobian matrix for the current step
  */
  const double
      px = x_(0),
      py = x_(1),
      vx = x_(2),
      vy = x_(3);

  // Convert state to measurement space
  const double
      rho = sqrt(px * px + py * py),
      phi = atan(py / px),
      rho_dot = (px * vx + py * vy) / rho;

  VectorXd h = VectorXd(3);
  h << rho, phi, rho_dot;

  VectorXd y = z - h;
  UpdateByResidual(z, y);
}

void KalmanFilter::UpdateByResidual(const VectorXd &z, VectorXd y) {
  /**
  * Performs the update steps which are identical for standard and extended Kalman Filters.
  */
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();

  x_ = x_ + K * y;
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
