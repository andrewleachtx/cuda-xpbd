#include "apbd/ShapeTwoCuboid.h"
#include "se3/lib.h"
#include "util.h"

namespace apbd {

ShapeTwoCuboid::ShapeTwoCuboid(const Eigen::Vector3f &sides1,
                               const Eigen::Vector3f &sides2, float halfDis,
                               float halfAngle) {
    cuboid1 = ShapeCuboid(sides1);
    cuboid2 = ShapeCuboid(sides2);

    E1 = Eigen::Matrix4f::Identity();
    E2 = Eigen::Matrix4f::Identity();

    Eigen::Vector3f R = se3::aaToMat(Eigen::Vector3f(0, 1, 0), halfAngle);

    E1.block<3, 3>(0, 0) = R;
    E1(0, 3) = -halfDis;

    E2.block<3, 3>(0, 0) = R.transpose();
    E2(0, 3) = halfDis;
}

// https://stackoverflow.com/a/37492287/27629759
Eigen::Matrix<float, 6, 1> ShapeTwoCuboid::computeInertia(float density) const {
    Eigen::Matrix<float, 6, 1> I1 =
        se3::inertiaCuboid(this->cuboid1.sides, density);
    Eigen::Matrix<float, 6, 1> I2 =
        se3::inertiaCuboid(this->cuboid2.sides, density);
    Eigen::Matrix<float, 6, 6> A1 = se3::Ad(se3::inv(this->E1));
    Eigen::Matrix<float, 6, 6> A2 = se3::Ad(se3::inv(this->E2));
    Eigen::Matrix<float, 6, 6> I = A1.transpose() * I1.asDiagonal() * A1 +
                                   A2.transpose() * I2.asDiagonal() * A2;

    return I.diagonal();
}

Eigen::Vector3f ShapeTwoCuboid::toCenterLocal(const Eigen::Matrix4f &E,
                                              const Eigen::Vector3f &xl) const {
    Eigen::Vector4f xlc = E * Eigen::Vector4f(xl(0), xl(1), xl(2), 1.0f);
    return xlc.block<3, 1>(0, 0);
}

bool ShapeTwoCuboid::broadphaseGround(const Eigen::Matrix4f &E,
                                      const Eigen::Matrix4f &Eg) const {
    bool flag1 = this->cuboid1.broadphaseGround(E * this->E1, Eg);
    bool flag2 = this->cuboid2.broadphaseGround(E * this->E2, Eg);

    // assuming matlab bool + bool is 0 + 1 aka logical or
    return flag1 | flag2;
}

/*
function cdata = narrowphaseGround(this,E,Eg)
    cdata1 = this.cuboid1.narrowphaseGround(E*this.E1,Eg);
    cdata2 = this.cuboid1.narrowphaseGround(E*this.E2,Eg);
    for i = 1: length(cdata1)
        cdata1(i).x1 = this.toCenterLocal(this.E1,cdata1(i).x1);
    end
    for i = 1: length(cdata2)
        cdata2(i).x1 = this.toCenterLocal(this.E2,cdata2(i).x1);
    end
    cdata = [cdata1 cdata2];
end
*/
cdata_t ShapeTwoCuboid::narrowphaseGround(const Eigen::Matrix4f &E,
                                          const Eigen::Matrix4f &Eg) const {
    cdata_t cdata1 = this->cuboid1.narrowphaseGround(E * this->E1, Eg);
    cdata_t cdata2 = this->cuboid2.narrowphaseGround(E * this->E2, Eg);
    for (size_t i = 0; i < cdata1.second; i++) {
        cdata1.first[i].x1 = this->toCenterLocal(this->E1, cdata1.first[i].x1);
    }
    for (size_t i = 0; i < cdata2.second; i++) {
        cdata2.first[i].x1 = this->toCenterLocal(this->E2, cdata2.first[i].x1);
    }

    // FIXME: Because we have a max of 8 contact points, we can merge them in.
    // Might be wrong approach, can look into this.
    cuda::std::array<Contact, 8> combined;
    size_t combined_ct = 0;

    for (size_t i = 0; i < cdata1.second && combined_ct < 8; i++) {
        combined[combined_ct++] = cdata1.first[i];
    }
    for (size_t i = 0; i < cdata2.second && combined_ct < 8; i++) {
        combined[combined_ct++] = cdata2.first[i];
    }

    return cdata_t(combined, combined_ct);
}

/*
function flag = broadphaseShape(this,E1,that,E2)
    if isa(that,'apbd.ShapeCuboid')
        flag1 = this.cuboid1.broadphaseShape(E1*this.E1,that,E2);
        flag2 = this.cuboid2.broadphaseShape(E1*this.E2,that,E2);
        flag = flag1 + flag2;
    elseif isa(that, 'apbd.ShapeTwoCuboid')
        flag1 =
this.cuboid1.broadphaseShape(E1*this.E1,that.cuboid1,E2*that.E1); flag2 =
this.cuboid1.broadphaseShape(E1*this.E1,that.cuboid2,E2*that.E2); flag3 =
this.cuboid2.broadphaseShape(E1*this.E2,that.cuboid1,E2*that.E1); flag4 =
this.cuboid2.broadphaseShape(E1*this.E2,that.cuboid2,E2*that.E2); flag = flag1 +
flag2 + flag3 + flag4; else error('Unsupported shape'); end end
*/

/*
TODO: Because of how this is designed, I cannot do inter-shape collisions,
      only intra. So for now we will not use this, and instead use the case
      statement to check for that in Shape.cu when a ShapeTwoCuboid collision
      is detected.
*/
bool ShapeTwoCuboid::broadphaseShape(const Eigen::Matrix4f &E1,
                                     const ShapeTwoCuboid &other,
                                     const Eigen::Matrix4f &E2) const {
    
}

/*
function cdata = narrowphaseShape(this,E1,that,E2)
    if isa(that,'apbd.ShapeCuboid')
        cdata1 = this.cuboid1.narrowphaseShape(E1*this.E1,that,E2);
        cdata2 = this.cuboid2.narrowphaseShape(E1*this.E2,that,E2);
        for i = 1: length(cdata1)
            cdata1(i).x1 = this.toCenterLocal(this.E1,cdata1(i).x1);
        end
        for i = 1: length(cdata2)
            cdata2(i).x1 = this.toCenterLocal(this.E2,cdata2(i).x1);
        end
        cdata = [cdata1 cdata2];
    elseif isa(that, 'apbd.ShapeTwoCuboid')
        cdata1 =
this.cuboid1.narrowphaseShape(E1*this.E1,that.cuboid1,E2*that.E1); cdata2 =
this.cuboid1.narrowphaseShape(E1*this.E1,that.cuboid2,E2*that.E2); cdata3 =
this.cuboid2.narrowphaseShape(E1*this.E2,that.cuboid1,E2*that.E1); cdata4 =
this.cuboid2.narrowphaseShape(E1*this.E2,that.cuboid2,E2*that.E2); for i = 1:
length(cdata1) cdata1(i).x1 = this.toCenterLocal(this.E1,cdata1(i).x1);
            cdata1(i).x2 = that.toCenterLocal(that.E1,cdata1(i).x2);
        end
        for i = 1: length(cdata2)
            cdata2(i).x1 = this.toCenterLocal(this.E1,cdata2(i).x1);
            cdata2(i).x2 = that.toCenterLocal(that.E2,cdata2(i).x2);
        end
        for i = 1: length(cdata3)
            cdata3(i).x1 = this.toCenterLocal(this.E2,cdata3(i).x1);
            cdata3(i).x2 = that.toCenterLocal(that.E1,cdata3(i).x2);
        end
        for i = 1: length(cdata4)
            cdata4(i).x1 = this.toCenterLocal(this.E2,cdata4(i).x1);
            cdata4(i).x2 = that.toCenterLocal(that.E2,cdata4(i).x2);
        end
        cdata = [cdata1 Ocdata2 cdata3 cdata4];
    else
        error('Unsupported shape');
    end
end
*/
/*
TODO: Because of how this is designed, I cannot do inter-shape collisions,
      only intra. So for now we will not use this, and instead use the case
      statement to check for that in Shape.cu when a ShapeTwoCuboid collision
      is detected.

      cdata_t ShapeTwoCuboid::narrowphaseShape(const Eigen::Matrix4f &E1,
                                              const ShapeTwoCuboid &other,
                                              const Eigen::Matrix4f &E2) const {
          
      }
*/

}  // namespace apbd