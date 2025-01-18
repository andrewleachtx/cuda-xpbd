#pragma once
#ifndef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#endif
#include <Eigen/Dense>

namespace apbd {

struct Contact {
    Eigen::Vector3f nw;
    Eigen::Vector3f x1;
    Eigen::Vector3f x2;
};

}  // namespace apbd
