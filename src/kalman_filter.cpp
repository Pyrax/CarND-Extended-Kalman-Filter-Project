#include "kalman_filter.h"
#include <math.h>
#include <iostream>

#define MIN_FLOAT_VALUE 0.000001

using namespace std;
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
  double
      px = x_(0),
      py = x_(1),
      vx = x_(2),
      vy = x_(3);

  if (fabs(px) < MIN_FLOAT_VALUE) {
    px = MIN_FLOAT_VALUE;
  }

  if (fabs(py) < MIN_FLOAT_VALUE) {
    py = MIN_FLOAT_VALUE;
  }

  // Convert state to measurement space
  double
      rho = sqrt(px * px + py * py),
      phi = atan2(py, px),
      rho_dot = (px * vx + py * vy) / rho;

  // cout << "phi: " << phi << endl;

  VectorXd h = VectorXd(3);
  h << rho, phi, rho_dot;

  VectorXd y = z - h;

  // Adjust resulting phi in y to be between -pi and pi
  while(y(1) < -M_PI || y(1) > M_PI) {
    y(1) += (y(1) < -M_PI) ? 2*M_PI : -2*M_PI;
  }

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
