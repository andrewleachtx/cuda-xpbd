#include "JGT-float/smits_mul.h"
#include "apbd/Shape.h"
#include "collideBoxBox/odeBoxBox.h"
#include "se3/lib.h"
#include "util.h"

namespace apbd {
Shape::Shape(ShapeCuboid cuboid) : type(SHAPE_CUBOID) { data.cuboid = cuboid; }
Shape::Shape(ShapeMeshObj meshObj) : type(SHAPE_MESHOBJ) {
    data.meshObj = new ShapeMeshObj(meshObj);
}
Shape::Shape(ShapeTwoCuboid twoCuboid) : type(SHAPE_TWOCUBOID) {
    data.twoCuboid = twoCuboid;
}

Shape::Shape(const Shape &other) : type(other.type) {
    switch (type) {
        case SHAPE_CUBOID: {
            data.cuboid = other.data.cuboid;
            break;
        }
        case SHAPE_MESHOBJ: {
            data.meshObj = new ShapeMeshObj(*other.data.meshObj);
            break;
        }
        case SHAPE_TWOCUBOID: {
            data.twoCuboid = other.data.twoCuboid;
            break;
        }
    }
}

Shape &Shape::operator=(const Shape &other) {
    if (this != &other) {
        type = other.type;
        switch (type) {
            case SHAPE_CUBOID: {
                data.cuboid = other.data.cuboid;
                break;
            }
            case SHAPE_MESHOBJ: {
                if (data.meshObj != nullptr) {
                    delete data.meshObj;
                    data.meshObj = nullptr;
                }
                data.meshObj = new ShapeMeshObj(*other.data.meshObj);
                break;
            }
            case SHAPE_TWOCUBOID: {
                data.twoCuboid = other.data.twoCuboid;
                break;
            }
        }
    }

    return *this;
}

bool Shape::broadphaseGround(Eigen::Matrix4f E, Eigen::Matrix4f Eg) const {
    switch (type) {
        case SHAPE_CUBOID: {
            return data.cuboid.broadphaseGround(E, Eg);
        }
        case SHAPE_MESHOBJ: {
            return data.meshObj->broadphaseGround(E, Eg);
        }
        case SHAPE_TWOCUBOID: {
            return data.twoCuboid.broadphaseGround(E, Eg);
        }
        default:
            return false;
    }
}

cdata_t Shape::narrowphaseGround(const Eigen::Matrix4f E,
                                 const Eigen::Matrix4f Eg) const {
    auto cdata = cuda::std::array<Contact, 8>();
    switch (type) {
        case SHAPE_CUBOID: {
            return data.cuboid.narrowphaseGround(E, Eg);
        }
        case SHAPE_MESHOBJ: {
            return data.meshObj->narrowphaseGround(E, Eg);
        }
        case SHAPE_TWOCUBOID: {
            return data.twoCuboid.narrowphaseGround(E, Eg);
        }
        default:
            return cdata_t(cdata, 0);
    }
}

/*
    Nested switch logic was removed for inter-subclass collisions
    (i.e. cuboid x bunny) checks, although pairs originating from
    shape_twocuboid are enabled for cuboids
    TODO: I'm guessing the inner switch should have been other.type as if you
   could have a bunny vs cube collision check, but we don't. This can be added
   back, but for simplicity I am removing it - only supporting same-type
   collisions.
*/
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
bool Shape::broadphaseShape(const Eigen::Matrix4f E1, const Shape &other,
                            const Eigen::Matrix4f E2) const {
    switch (type) {
        case SHAPE_CUBOID: {
            return this->data.cuboid.broadphaseShapeCuboid(
                E1, other.data.cuboid, E2);
        }
        case SHAPE_MESHOBJ: {
            return this->data.meshObj->broadphaseShapeMesh(
                E1, *other.data.meshObj, E2);
        }
        case SHAPE_TWOCUBOID: {
            switch (other.type) {
                case SHAPE_CUBOID: {
                    const auto &c1 = this->data.twoCuboid.cuboid1;
                    const auto &c2 = this->data.twoCuboid.cuboid2;

                    bool flag1 = c1.broadphaseShapeCuboid(
                        E1 * this->data.twoCuboid.E1, other.data.cuboid, E2);
                    bool flag2 = c2.broadphaseShapeCuboid(
                        E1 * this->data.twoCuboid.E2, other.data.cuboid, E2);

                    return flag1 | flag2;
                }
                case SHAPE_TWOCUBOID: {
                    const auto &c1 = this->data.twoCuboid.cuboid1;
                    const auto &c2 = this->data.twoCuboid.cuboid2;

                    const auto E1_thisE1 = E1 * this->data.twoCuboid.E1;
                    const auto E1_thisE2 = E1 * this->data.twoCuboid.E2;
                    const auto E2_thatE1 = E2 * other.data.twoCuboid.E1;
                    const auto E2_thatE2 = E2 * other.data.twoCuboid.E2;

                    bool flag1 = c1.broadphaseShapeCuboid(
                        E1_thisE1, other.data.twoCuboid.cuboid1, E2_thatE1);
                    bool flag2 = c1.broadphaseShapeCuboid(
                        E1_thisE1, other.data.twoCuboid.cuboid2, E2_thatE2);
                    bool flag3 = c2.broadphaseShapeCuboid(
                        E1_thisE2, other.data.twoCuboid.cuboid1, E2_thatE1);
                    bool flag4 = c2.broadphaseShapeCuboid(
                        E1_thisE2, other.data.twoCuboid.cuboid2, E2_thatE2);

                    return flag1 | flag2 | flag3 | flag4;
                }
                default: {
                    return false;
                }
            }
        }

        default:
            return false;
    }
}

// See message in broadphaseShape above.
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
cdata_t Shape::narrowphaseShape(const Eigen::Matrix4f E1, const Shape &other,
                                const Eigen::Matrix4f E2) const {
    switch (type) {
        case SHAPE_CUBOID: {
            return this->data.cuboid.narrowphaseShapeCuboid(
                E1, other.data.cuboid, E2);
        }
        case SHAPE_MESHOBJ: {
            return this->data.meshObj->narrowphaseShapeMesh(
                E1, *other.data.meshObj, E2);
        }
        case SHAPE_TWOCUBOID: {
            switch (other.type) {
                case SHAPE_CUBOID: {
                    auto cdata1 =
                        this->data.twoCuboid.cuboid1.narrowphaseShapeCuboid(
                            E1 * this->data.twoCuboid.E1, other.data.cuboid,
                            E2);
                    auto cdata2 =
                        this->data.twoCuboid.cuboid2.narrowphaseShapeCuboid(
                            E1 * this->data.twoCuboid.E2, other.data.cuboid,
                            E2);

                    for (size_t i = 0; i < cdata1.second; i++) {
                        cdata1.first[i].x1 = this->data.twoCuboid.toCenterLocal(
                            this->data.twoCuboid.E1, cdata1.first[i].x1);
                    }
                    for (size_t i = 0; i < cdata2.second; i++) {
                        cdata2.first[i].x1 = this->data.twoCuboid.toCenterLocal(
                            this->data.twoCuboid.E2, cdata2.first[i].x1);
                    }

                    // FIXME: Same combine logic as in ShapeTwoCuboid.h, might
                    // need to check this
                    cuda::std::array<Contact, 8> combined;
                    size_t combined_ct = 0;

                    for (size_t i = 0; i < cdata1.second && combined_ct < 8;
                         i++) {
                        combined[combined_ct++] = cdata1.first[i];
                    }
                    for (size_t i = 0; i < cdata2.second && combined_ct < 8;
                         i++) {
                        combined[combined_ct++] = cdata2.first[i];
                    }

                    return cdata_t(combined, combined_ct);
                }
                case SHAPE_TWOCUBOID: {
                    const auto E1_thisE1 = E1 * this->data.twoCuboid.E1;
                    const auto E1_thisE2 = E1 * this->data.twoCuboid.E2;
                    const auto E2_thatE1 = E2 * other.data.twoCuboid.E1;
                    const auto E2_thatE2 = E2 * other.data.twoCuboid.E2;

                    auto cdata1 =
                        this->data.twoCuboid.cuboid1.narrowphaseShapeCuboid(
                            E1_thisE1, other.data.twoCuboid.cuboid1, E2_thatE1);
                    auto cdata2 =
                        this->data.twoCuboid.cuboid1.narrowphaseShapeCuboid(
                            E1_thisE1, other.data.twoCuboid.cuboid2, E2_thatE2);
                    auto cdata3 =
                        this->data.twoCuboid.cuboid2.narrowphaseShapeCuboid(
                            E1_thisE2, other.data.twoCuboid.cuboid1, E2_thatE1);
                    auto cdata4 =
                        this->data.twoCuboid.cuboid2.narrowphaseShapeCuboid(
                            E1_thisE2, other.data.twoCuboid.cuboid2, E2_thatE2);

                    for (size_t i = 0; i < cdata1.second; i++) {
                        cdata1.first[i].x1 = this->data.twoCuboid.toCenterLocal(
                            this->data.twoCuboid.E1, cdata1.first[i].x1);
                        cdata1.first[i].x2 = other.data.twoCuboid.toCenterLocal(
                            other.data.twoCuboid.E1, cdata1.first[i].x2);
                    }
                    for (size_t i = 0; i < cdata2.second; i++) {
                        cdata2.first[i].x1 = this->data.twoCuboid.toCenterLocal(
                            this->data.twoCuboid.E1, cdata2.first[i].x1);
                        cdata2.first[i].x2 = other.data.twoCuboid.toCenterLocal(
                            other.data.twoCuboid.E2, cdata2.first[i].x2);
                    }
                    for (size_t i = 0; i < cdata3.second; i++) {
                        cdata3.first[i].x1 = this->data.twoCuboid.toCenterLocal(
                            this->data.twoCuboid.E2, cdata3.first[i].x1);
                        cdata3.first[i].x2 = other.data.twoCuboid.toCenterLocal(
                            other.data.twoCuboid.E1, cdata3.first[i].x2);
                    }
                    for (size_t i = 0; i < cdata4.second; i++) {
                        cdata4.first[i].x1 = this->data.twoCuboid.toCenterLocal(
                            this->data.twoCuboid.E2, cdata4.first[i].x1);
                        cdata4.first[i].x2 = other.data.twoCuboid.toCenterLocal(
                            other.data.twoCuboid.E2, cdata4.first[i].x2);
                    }

                    // FIXME: Again, "cdata = [cdata1 cdata2 cdata3 cdata4];"
                    // suggests combination logic, so I will do that that said,
                    // there is a max of 8 contacts, so this may need
                    // refactoring
                    cuda::std::array<Contact, 8> combined;
                    size_t combined_ct = 0;

                    for (size_t i = 0; i < cdata1.second && combined_ct < 8;
                         i++) {
                        combined[combined_ct++] = cdata1.first[i];
                    }
                    for (size_t i = 0; i < cdata2.second && combined_ct < 8;
                         i++) {
                        combined[combined_ct++] = cdata2.first[i];
                    }
                    for (size_t i = 0; i < cdata3.second && combined_ct < 8;
                         i++) {
                        combined[combined_ct++] = cdata3.first[i];
                    }
                    for (size_t i = 0; i < cdata4.second && combined_ct < 8;
                         i++) {
                        combined[combined_ct++] = cdata4.first[i];
                    }

                    return cdata_t(combined, combined_ct);
                }
            }
        }
        default: {
            return cdata_t(cuda::std::array<Contact, 8>(), 0);
        }
    }
}

Eigen::Matrix<float, 6, 1> Shape::computeInertia(const float density) const {
    switch (type) {
        case SHAPE_CUBOID: {
            // I is the 6x1 diagonal rigid inertia, assuming that the frame
            // origin is at the center of mass and the axes are oriented along
            // the principal axes. We store the rotation on top of translations,
            // so that I(1:3) is the rotational inertia and I(4:6) is the
            // translational inertia.
            return se3::inertiaCuboid(this->data.cuboid.sides, density);
        }
        case SHAPE_MESHOBJ: {
            return data.meshObj->computeInertia(density);
        }
        case SHAPE_TWOCUBOID: {
            return data.twoCuboid.computeInertia(density);
        }
        default:
            return Eigen::Matrix<float, 6, 1>();
    }
}

}  // namespace apbd
