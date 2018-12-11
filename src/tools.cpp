#include <iostream>
#include <assert.h>
#include "tools.h"

#define TOTAL_STATE_VARIABLES 4

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  assert(!estimations.empty());  // calculating RMSE without elements would lead to division by zero
  assert(estimations.size() == ground_truth.size());

  unsigned long n = estimations.size();
  VectorXd rmse = VectorXd::Zero(TOTAL_STATE_VARIABLES);

  for (int i = 0; i < n; i++) {
    // calculate squared residual component-wise for each vector in the list
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  rmse /= n;
  return rmse.array().sqrt();
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state) {
  MatrixXd Hj(3, 4);

  const double
      px = x_state(0),
      py = x_state(1),
      vx = x_state(2),
      vy = x_state(3);

  // pre-compute a set of terms used several times in Jacobian
  const double
      c1 = px * px + py * py,
      c2 = sqrt(c1),
      c3 = c2 * c1;

  // ensure that c1 is not zero, otherwise terms will lead to division by zero
  assert(fabs(c1) > 0.00001);

  // compute Jacobian matrix
  Hj <<
     px / c2, py / c2, 0, 0,
      -py / c1, px / c1, 0, 0,
      py * (vx * py - vy * px) / c3, px * (px * vy - py * vx) / c3, px / c2, py / c2;

  return Hj;
}
