#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  // Create return vector
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // Check if there are estimations and there are as many as ground truths
  assert(estimations.size() != 0);
  assert(estimations.size() == ground_truth.size());

  // Calculate residuals
  for (int i=0; i < estimations.size(); ++i){
    VectorXd residual = estimations[i] - ground_truth[i];

    residual = residual.array() * residual.array();
    rmse += residual;
  }

  // Calculate the mean
  rmse = rmse / estimations.size();

  // Square Root
  rmse = rmse.array().sqrt();

  return rmse;
  }

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  // Create return Matrix
  MatrixXd Hj(3,4);

  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // Optimize with precomputed coefficients
  float c1 = px * px + py * py;
  float c2 = std::sqrt(c1);
  float c3 = c1 * c2;

  // catch division with 0
  if (c2 == 0) {
    std::cout << "Error : Division by zero" << std::endl;
    return Hj;
  }

  Hj << (px / c2), (py / c2), 0, 0,
        -(py / c1), (px / c1), 0, 0,
        py * (vx * py - vy * px)/c3, px * (px * vy - py * vx) / c3, px / c2, py / c2;

  return Hj;

}
