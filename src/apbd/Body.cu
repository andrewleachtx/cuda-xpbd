#include <svd3_cuda.h>

#include <limits>

#include "apbd/Body.h"
#include "se3/lib.h"
#include "util.h"

namespace apbd {

using Eigen::Vector3f;

Body::Body() : type(BODY_INVALID), data{0} {}
Body::Body(BodyRigid rigid) : type(BODY_RIGID), data{.rigid{rigid}} {}
Body::Body(BodyAffine affine)
    : type(BODY_AFFINE), data{.affine = std::move(affine)} {}
Body &Body::operator=(const Body &&other) {
    this->type = other.type;
    switch (type) {
        case BODY_AFFINE:
            this->data.affine = std::move(other.data.affine);
            break;
        case BODY_RIGID:
            this->data.rigid = std::move(other.data.rigid);
            break;
        default:
            break;
    }
    return *this;
}

void Body::setInitTransform(Eigen::Matrix4f E) {
    switch (this->type) {
        case BODY_AFFINE: {
            auto &data = this->data.affine;
            data.xInit.block<9, 1>(0, 0) = E.block<3, 3>(0, 0).reshaped(9, 1);
            data.xInit.block<3, 1>(9, 0) = E.block<3, 1>(0, 3);
            break;
        }
        case BODY_RIGID: {
            auto &data = this->data.rigid;
            data.xInit.block<4, 1>(0, 0) =
                Eigen::Quaternionf(E.block<3, 3>(0, 0)).coeffs();
            if (data.xInit(3) < 0) {
                data.xInit = -data.xInit;
            }
            data.xInit.block<3, 1>(4, 0) = E.block<3, 1>(0, 3);
            break;
        }
        default:
            break;
    }
}

void Body::setInitVelocity(Eigen::Matrix<float, 6, 1> velocity) {
    switch (this->type) {
        case BODY_AFFINE: {
            break;
        }
        case BODY_RIGID: {
            auto &data = this->data.rigid;
            Eigen::Quaternionf q =
                Eigen::Quaternionf(data.xInit.block<4, 1>(0, 0));
            data.v = velocity.block<3, 1>(3, 0);
            data.w = velocity.block<3, 1>(0, 0);
            break;
        }
        default:
            break;
    }
}

void Body::clearJacobi() {
    switch (this->type) {
        case BODY_AFFINE: {
            break;
        }
        case BODY_RIGID: {
            auto &data = this->data.rigid;
            data.dxJacobi = vec7::Zero();
        }
        default:
            break;
    }
}

BodyRigid::BodyRigid(Shape shape, float density)
    : xInit(vec7::Zero()),
      x(vec7::Zero()),
      x0(vec7::Zero()),
      collide(false),
      mu(0.0),
      layer(99),
      shape(shape),
      density(density),
      Mr(Eigen::Vector3f::Zero()),
      Mp(0),
      v(Vector3f::Zero()),
      w(Vector3f::Zero()),
      dxJacobi(vec7::Zero()),
      dphiJacobi(vec7::Zero()) {}
BodyRigid::BodyRigid(Shape shape, float density, bool collide, float mu)
    : xInit(vec7::Zero()),
      x(vec7::Zero()),
      x0(vec7::Zero()),
      collide(collide),
      mu(mu),
      layer(99),
      shape(shape),
      density(density),
      Mr(Eigen::Vector3f::Zero()),
      Mp(0),
      v(Vector3f::Zero()),
      w(Vector3f::Zero()),
      dxJacobi(vec7::Zero()),
      dphiJacobi(vec7::Zero()) {}

}  // namespace apbd
