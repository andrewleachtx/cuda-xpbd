#include "JGT-float/smits_mul.h"
#include "apbd/Shape.h"
#include "collideBoxBox/odeBoxBox.h"
#include "se3/lib.h"
#include "util.h"

namespace apbd {

Shape::Shape(ShapeCuboid cuboid) : type(SHAPE_CUBOID), data{.cuboid = cuboid} {}
Shape &Shape::operator=(const Shape &other) {
    this->type = other.type;
    switch (type) {
        case SHAPE_CUBOID: {
            this->data.cuboid = other.data.cuboid;
        }
        default:
            break;
    }
    return *this;
}

bool Shape::broadphaseGround(Eigen::Matrix4f E, Eigen::Matrix4f Eg) const {
    switch (type) {
        case SHAPE_CUBOID: {
            auto data = this->data.cuboid;

            // Check the height of the center
            Eigen::Vector4f xl(0.0, 0.0, 0.0, 1.0);
            Eigen::Vector4f xw = E * xl;
            Eigen::Vector4f xg = Eg.inverse() * xw;
            float r = (data.sides / 2).norm();  // dist to a corner
            return xg(2) < 1.5 * r;
        }
        default:
            return false;
    }
}

cuda::std::pair<cuda::std::array<Contact, 8>, size_t> Shape::narrowphaseGround(
    const Eigen::Matrix4f E, const Eigen::Matrix4f Eg) const {
    auto cdata = cuda::std::array<Contact, 8>();
    switch (type) {
        case SHAPE_CUBOID: {
            const auto data = this->data.cuboid;

            const Eigen::Vector3f s = data.sides / 2;
            // local space
            Eigen::Matrix<float, 4, 8> xl = Eigen::Matrix<float, 4, 8>::Ones();
            xl.block<3, 1>(0, 0) = Eigen::Vector3f(-s(0), s(1), -s(2));
            xl.block<3, 1>(0, 1) = Eigen::Vector3f(-s(0), -s(1), -s(2));
            xl.block<3, 1>(0, 2) = Eigen::Vector3f(s(0), -s(1), -s(2));
            xl.block<3, 1>(0, 3) = Eigen::Vector3f(s(0), s(1), -s(2));
            xl.block<3, 1>(0, 4) = Eigen::Vector3f(-s(0), s(1), s(2));
            xl.block<3, 1>(0, 5) = Eigen::Vector3f(-s(0), -s(1), s(2));
            xl.block<3, 1>(0, 6) = Eigen::Vector3f(s(0), -s(1), s(2));
            xl.block<3, 1>(0, 7) = Eigen::Vector3f(s(0), s(1), s(2));

            const Eigen::Matrix<float, 4, 8> xw = E * xl;
            const Eigen::Matrix<float, 4, 8> xg = Eg.inverse() * xw;

            int cdata_count = 0;
            for (size_t i = 0; i < 8; i++) {
                // This only supports vertex collisions
                const float d = xg(2, i);
                if (d < 0.2) {
                    Eigen::Vector4f xgproj = xg.block<4, 1>(0, i);
                    // project onto the floor plane
                    xgproj(2) = 0;
                    cdata[cdata_count++] = Contact{
                        // normal
                        .nw = Eg.block<3, 1>(0, 2),
                        .x1 = xl.block<3, 1>(0, i),
                        // transform to world space
                        .x2 = Eg.block<3, 4>(0, 0) * xgproj,
                    };
                }
            }

            return cuda::std::pair<cuda::std::array<Contact, 8>, size_t>(
                cdata, cdata_count);
        }
        default:
            return cuda::std::pair<cuda::std::array<Contact, 8>, size_t>(cdata,
                                                                         0);
    }
}

bool Shape::broadphaseShape(const Eigen::Matrix4f E1, const Shape &other,
                            const Eigen::Matrix4f E2) const {
    switch (type) {
        case SHAPE_CUBOID:
            switch (type) {
                case SHAPE_CUBOID:
                    return this->data.cuboid.broadphaseShapeCuboid(
                        E1, other.data.cuboid, E2);
                default:
                    return false;
            }
        default:
            return false;
    }
}

cuda::std::pair<cuda::std::array<Contact, 8>, size_t> Shape::narrowphaseShape(
    const Eigen::Matrix4f E1, const Shape &other,
    const Eigen::Matrix4f E2) const {
    switch (type) {
        case SHAPE_CUBOID:
            switch (type) {
                case SHAPE_CUBOID:
                    return this->data.cuboid.narrowphaseShapeCuboid(
                        E1, other.data.cuboid, E2);
                default:
                    return cuda::std::pair<cuda::std::array<Contact, 8>,
                                           size_t>(
                        cuda::std::array<Contact, 8>(), 0);
            }
        default:
            return cuda::std::pair<cuda::std::array<Contact, 8>, size_t>(
                cuda::std::array<Contact, 8>(), 0);
    }
}

bool ShapeCuboid::broadphaseShapeCuboid(const Eigen::Matrix4f E1,
                                        const ShapeCuboid &other,
                                        const Eigen::Matrix4f E2) const {
    Eigen::Vector3f p1 = E1.block<3, 1>(0, 3);
    Eigen::Vector3f p2 = E2.block<3, 1>(0, 3);
    float d = (p1 - p2).norm();
    float r1 = (this->sides / 2).norm();  // dist to a corner
    float r2 = (other.sides / 2).norm();  // dist to a corner
    return d <= 1.5 * (r1 + r2);
}

cuda::std::pair<cuda::std::array<Contact, 8>, size_t>
ShapeCuboid::narrowphaseShapeCuboid(const Eigen::Matrix4f E1,
                                    const ShapeCuboid &other,
                                    const Eigen::Matrix4f E2) const {
    cuda::std::array<Contact, 8> cdata{};
    const Eigen::Matrix3f R1 = E1.block<3, 3>(0, 0);
    const Eigen::Matrix3f R2 = E2.block<3, 3>(0, 0);
    const Eigen::Vector3f p1 = E1.block<3, 1>(0, 3);
    const Eigen::Vector3f p2 = E2.block<3, 1>(0, 3);
    const auto &s1 = this->sides;
    const auto &s2 = other.sides;
    const auto collisions = odeBoxBox(E1, s1, E2, s2);
    const Eigen::Vector3f nw =
        collisions.normal;  // The normal is outward from body 1 (red)
    const Eigen::Vector3f n1 = R1.transpose() * nw;
    const Eigen::Vector3f n2 =
        -R2.transpose() * nw;  // negate since nw is defined wrt body 1
    for (size_t i = 0; i < collisions.count && i < 8; i++) {
        const Eigen::Vector3f xw = collisions.positions[i];
        Eigen::Vector3f x1 = R1.transpose() * (xw - p1);
        const float t1 = this->raycast(x1, n1);

        // negate since smits_mul returns negative t for rays starting
        // inside the box Compute local point on body 2 with ray casting
        x1 = x1 - t1 * n1;

        Eigen::Vector3f x2 = R2.transpose() * (xw - p2);
        const float t2 = other.raycast(x2, n2);
        x2 = x2 - t2 * n2;  // negate since smits_mul returns negative t for
                            // rays starting inside the box
        cdata[i] = Contact{.nw = nw, .x1 = x1, .x2 = x2};
    }
    return cuda::std::pair<cuda::std::array<Contact, 8>, size_t>(
        cdata, collisions.count);
}

float ShapeCuboid::raycast(Eigen::Vector3f x, Eigen::Vector3f n) const {
    const float thresh = 1e-9;
    const Eigen::Vector3f bmax = 0.5 * this->sides;
    const Eigen::Vector3f bmin = -bmax;
    x = (1 - thresh) * x;  // make the point go slightly inside the box
    n = -n;                // negate ray since it starts inside the box
    jgt_float::ray r{0};
    jgt_float::make_ray(x(0), x(1), x(2), n(0), n(1), n(2), &r);
    jgt_float::aabox a{0};
    jgt_float::make_aabox(bmin(0), bmin(1), bmin(2), bmax(0), bmax(1), bmax(2),
                          &a);
    float t = 0;
    const bool _hit = jgt_float::smits_mul(&r, &a, &t);
    return t;
}

Eigen::Matrix<float, 6, 1> Shape::computeInertia(const float density) const {
    switch (type) {
        case SHAPE_CUBOID:
            switch (type) {
                case SHAPE_CUBOID:
                    // I is the 6x1 diagonal rigid inertia, assuming that the
                    // frame origin is at the center of mass and the axes are
                    // oriented along the principal axes. We store the rotation
                    // on top of translations, so that I(1:3) is the rotational
                    // inertia and I(4:6) is the translational inertia.
                    return se3::inertiaCuboid(this->data.cuboid.sides, density);
                default:
                    return Eigen::Matrix<float, 6, 1>();
            }
        default:
            return Eigen::Matrix<float, 6, 1>();
    }
}

}  // namespace apbd
