#pragma once
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include <Eigen/Dense>

namespace apbd {

struct Contact {
  Eigen::Vector3f nw;
  Eigen::Vector3f x1;
  Eigen::Vector3f x2;
};

} // namespace apbd
