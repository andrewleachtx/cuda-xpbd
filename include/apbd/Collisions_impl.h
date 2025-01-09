#pragma once
#include "apbd/BodyReference_impl.h"
#include "apbd/Collisions.h"
#include "apbd/ConstraintReference_impl.h"

namespace apbd {

inline void Collision::setContacts(CollisionReference clr,
                                   const cdata_t cdata) {
    this->contacts = cdata.first;
    clr.contactNum(cdata.second);

    clr.J1I(mat24x6f::Zero());
    clr.J2I(mat24x6f::Zero());
    clr.b(vec24f::Zero());
    clr.mu(0.0f);

    clr.lambda(vec24f::Zero());
    clr.lambdac(vec24f::Zero());
    clr.lambdad(vec24f::Zero());

    clr.t_bar(vec24f::Zero());
    clr.g(vec24f::Zero());
    clr.p(vec24f::Zero());
    clr.Ax(vec24f::Zero());
    clr.freeIndex(vec24b::Zero());

    clr.J1I_cg(mat24x6f::Zero());
    clr.J2I_cg(mat24x6f::Zero());
    clr.r_cg(vec24f::Zero());
    clr.b_cg(vec24f::Zero());
    clr.Minv_cg(vec24f::Zero());
    clr.g_cg(vec24f::Zero());
    clr.d_cg(vec24f::Zero());
}

inline bool Collision::is_ground(CollisionReference clr) {
    return clr.body2() == NULL_BODY;
}

inline void Collision::getConstraints(CollisionReference clr,
                                      size_t& ground_count,
                                      size_t& rigid_count) {
    if (this->is_ground(clr)) {
        for (unsigned int i = 0; i < clr.contactNum(); i++) {
            ConstraintReference cr(ground_count++);
            // TODO: handle other body types
            cr.get_ground().create(clr.body1().get_rigid(), contacts[i], clr);
            this->constraints[i] = cr;
        }
    } else {
        for (unsigned int i = 0; i < clr.contactNum(); i++) {
            ConstraintReference cr(rigid_count++);
            cr.get_rigid().create(clr.body1().get_rigid(),
                                  clr.body2().get_rigid(), contacts[i], clr);
            this->constraints[i] = cr;
        }
    }
}

inline void Collision::solveCollisionNor(CollisionReference clr, float hs,
                                         float biasCoeff, float minpenetration,
                                         bool withSP) {
    for (unsigned int i = 0; i < clr.contactNum(); i++) {
        if (this->is_ground(clr)) {
            this->constraints[i].get_ground().solveNorPos(hs, biasCoeff,
                                                          minpenetration);
        } else {
            this->constraints[i].get_rigid().solveNorPos(
                hs, biasCoeff, minpenetration, withSP);
        }
    }
}

inline void Collision::solveCollisionTan(CollisionReference clr, float hs,
                                         float biasCoeff, bool withSP) {
    for (unsigned int i = 0; i < clr.contactNum(); i++) {
        if (this->is_ground(clr)) {
            this->constraints[i].get_ground().solveTanVel(hs, biasCoeff);
        } else {
            this->constraints[i].get_rigid().solveTanVel(hs, biasCoeff, withSP);
        }
    }
}

inline void Collision::applyLambdaSP(CollisionReference clr) {
    for (unsigned int i = 0; i < clr.contactNum(); i++) {
        if (this->is_ground(clr)) {
            this->constraints[i].get_ground().applyLambdaSP();
        } else {
            this->constraints[i].get_rigid().applyLambdaSP();
        }
    }
}

inline void Collision::initConstraints(CollisionReference clr) {
    for (unsigned int i = 0; i < clr.contactNum(); i++) {
        if (this->is_ground(clr)) {
            this->constraints[i].get_ground().init();
        } else {
            this->constraints[i].get_rigid().init();
        }
    }
}

inline void Collision::computeJ_b(CollisionReference clr, float h)
{
    unsigned int n = clr.contactNum();

    auto J1I_cpy = clr.J1I();
    auto J2I_cpy = clr.J2I();
    auto b_cpy   = clr.b();

    if (this->is_ground(clr))
    {
        for (unsigned int i = 0; i < n; i++)
        {
            unsigned int rows = 3 * i;
            
            auto grd = this->constraints[i].get_ground();

            const Eigen::Matrix3f &raXnI   = grd.raXnI1();
            const Eigen::Matrix3f &cFrame = grd.contactFrame();
            float mp = clr.body1().get_rigid().Mp();

            Eigen::Vector3f Cs = grd.evalCs(h);

            J1I_cpy.block<3,3>(rows, 0) = raXnI.transpose();
            J1I_cpy.block<3,3>(rows, 3) = cFrame.transpose() / std::sqrt(mp);

            b_cpy.block<3,1>(rows, 0)   = -Cs;

            bool zero_norm = Cs.norm() < 0.01f;
        }
    }
    else
    {
        for (unsigned int i = 0; i < n; i++)
        {
            unsigned int rows = 3 * i;

            auto rig = this->constraints[i].get_rigid();

            const Eigen::Matrix3f &raXnI1  = rig.raXnI1();
            const Eigen::Matrix3f &cFrame1 = rig.contactFrame();
            float mp1 = rig.body1().Mp();

            const Eigen::Matrix3f &raXnI2  = rig.raXnI2();
            const Eigen::Matrix3f &cFrame2 = rig.contactFrame();
            float mp2 = rig.body2().Mp();

            Eigen::Vector3f Cs = rig.evalCs(h);

            J1I_cpy.block<3,3>(rows, 0) = raXnI1.transpose();
            J1I_cpy.block<3,3>(rows, 3) = cFrame1.transpose() / std::sqrt(mp1);

            J2I_cpy.block<3,3>(rows, 0) = -raXnI2.transpose();
            J2I_cpy.block<3,3>(rows, 3) = -cFrame2.transpose() / std::sqrt(mp2);

            b_cpy.block<3,1>(rows, 0) = -Cs;
        }
    }

    clr.J1I(J1I_cpy);
    clr.J2I(J2I_cpy);
    clr.b(b_cpy);
}


// inline void Collision::computeJ_b(CollisionReference clr, float h) {
//     unsigned int n = clr.contactNum();

//     if (this->is_ground(clr)) {
//         for (unsigned int i = 0; i < n; i++) {
//             unsigned int rows = 3 * i;
//             auto grd = this->constraints[i].get_ground();

//             const Eigen::Matrix3f& raXnI = grd.raXnI1();
//             const Eigen::Matrix3f& c_frame = gr .contactFrame();
//             // TODO: Is it clr.body1() or grd.body() ?
//             float b1_Mp = clr.body1().get_rigid().Mp();

//             auto J1I_cpy = clr.J1I();
//             auto b_cpy = clr.b();

//             J1I_cpy.block<3, 3>(rows, 0) = raXnI.transpose();
//             J1I_cpy.block<3, 3>(rows, 3) = c_frame.transpose() / sqrt(b1_Mp);
//             b_cpy.block<3, 1>(rows, 0) = -grd.evalCs(h);

//             clr.J1I(J1I_cpy);
//             clr.b(b_cpy);
//         }
//     } else {
//         for (unsigned int i = 0; i < n; i++) {
//             unsigned int rows = 3 * i;

//             auto rig = this->constraints[i].get_rigid();

//             const Eigen::Matrix3f& raXnI1 = rig.raXnI1();
//             const Eigen::Matrix3f& c_frame1 = rig.contactFrame();
//             float b1_Mp = rig.body1().Mp();

//             const Eigen::Matrix3f& raXnI2 = rig.raXnI2();
//             const Eigen::Matrix3f& c_frame2 = rig.contactFrame();
//             float b2_Mp = rig.body2().Mp();

//             auto J1I_cpy = clr.J1I();
//             auto J2I_cpy = clr.J2I();
//             auto b_cpy = clr.b();

//             J1I_cpy.block<3, 3>(rows, 0) = raXnI1.transpose();
//             J1I_cpy.block<3, 3>(rows, 3) = c_frame1.transpose() /
//             sqrt(b1_Mp);

//             J2I_cpy.block<3, 3>(rows, 0) = -raXnI2.transpose();
//             J2I_cpy.block<3, 3>(rows, 3) = -c_frame2.transpose() /
//             sqrt(b2_Mp);

//             b_cpy.block<3, 1>(rows, 0) = -rig.evalCs(h);

//             clr.J1I(J1I_cpy);
//             clr.J2I(J2I_cpy);
//             clr.b(b_cpy);
//         }
//     }
// }

/*
function compute_LTlambda(this)
    this.body1.LTx = this.body1.LTx + this.J1I' * this.lambda;
    this.body2.LTx = this.body2.LTx + this.J2I' * this.lambda;
end
*/
inline void Collision::compute_LTlambda(CollisionReference clr) {
    // getters probably still make the copy even with const T&
    auto b1 = clr.body1().get_rigid();
    auto b2 = clr.body2().get_rigid();

    auto J1I = clr.J1I();
    auto J2I = clr.J2I();
    auto lambda = clr.lambda();
    
    b1.LTx(b1.LTx() + J1I.transpose() * lambda);

    // TODO: Optimize (see Collision::compute_LTp)
    if (!is_ground(clr)) {
        b2.LTx(b2.LTx() + J2I.transpose() * lambda);
    }

    if (b1.LTx().hasNaN()) {
        TRACE("b1.LTx has NaN");
        exit(1);
    }
    if (b2.LTx().hasNaN()) {
        TRACE("b2.LTx has NaN");
        exit(1);
    }
}

/*
function compute_degenerate_J1I_J2I_b(this)
    this.J1I_cg = this.J1I;
    this.J2I_cg = this.J2I;
    this.b_cg = this.b;
    for i = 1:3:3*this.contactNum
        if(this.freeIndex(i) && ~this.freeIndex(i+1))
            this.J1I_cg(i,:) = this.lambdad(i:i+2)' * this.J1I(i:i+2,:);
            this.J2I_cg(i,:) = this.lambdad(i:i+2)' * this.J2I(i:i+2,:);
            this.b_cg(i) = this.lambdad(i:i+2)' * this.b(i:i+2);
            this.lambda(i) = this.lambdad(i:i+2)' * this.lambda(i:i+2);
        end
    end
    this.Minv_cg = ones(3*this.contactNum,1);
end
*/
// FIXME: Debug this against MATLAB
inline void Collision::compute_degenerate_J1I_J2I_b(CollisionReference clr) {
    auto J1I_cgCpy = clr.J1I();
    auto J2I_cgCpy = clr.J2I();
    auto b_cgCpy = clr.b();
    auto fIdx_cpy = clr.freeIndex();
    auto lmd_cpy = clr.lambdad();
    auto lm_cpy = clr.lambda();

    const unsigned int n = 3 * clr.contactNum();
    for (unsigned int i = 0; i < n; i+= 3) {
        if (fIdx_cpy(i) && !fIdx_cpy(i + 1)) {
            auto lmd_seg = lmd_cpy.segment<3>(i);

            J1I_cgCpy.row(i) = lmd_seg.transpose() * clr.J1I().block<3, 6>(i, 0);
            J2I_cgCpy.row(i) = lmd_seg.transpose() * clr.J2I().block<3, 6>(i, 0);
            b_cgCpy(i) = lmd_seg.transpose() * clr.b().segment<3>(i);
            lm_cpy(i) = lmd_seg.transpose() * lm_cpy.segment<3>(i);
        }
    }

    clr.J1I_cg(J1I_cgCpy);
    clr.J2I_cg(J2I_cgCpy);
    clr.b_cg(b_cgCpy);
    clr.lambda(lm_cpy);

    if (clr.J1I_cg().hasNaN()) {
        TRACE("J1I_cg has NaN");
        exit(1);
    }
    if (clr.J2I_cg().hasNaN()) {
        TRACE("J2I_cg has NaN");
        exit(1);
    }
    if (clr.b_cg().hasNaN()) {
        TRACE("b_cg has NaN");
        exit(1);
    }
    if (clr.lambda().hasNaN()) {
        TRACE("lambda has NaN");
        exit(1);
    }

    clr.Minv_cg(vec24f::Ones());
}
// inline void Collision::compute_degenerate_J1I_J2I_b(CollisionReference clr) {
//     auto J1I_cpy = clr.J1I();
//     auto J2I_cpy = clr.J2I();
//     auto b_cpy = clr.b();
//     auto lm_cpy = clr.lambda();
//     auto lmd_cpy = clr.lambdad();
//     auto fidx_cpy = clr.freeIndex();

//     auto J1I_cgCpy = J1I_cpy;
//     auto J2I_cgCpy = J2I_cpy;
//     auto b_cgCpy = b_cpy;

//     // May need to support "collapse" semantics in MATLAB which would
//     // zero other indices
//     unsigned int n = 3 * clr.contactNum();
//     for (unsigned int i = 0; i < n; i += 3) {
//         if (fidx_cpy(i) != 0 && fidx_cpy(i + 1) == 0) {
//             Eigen::Vector3f lm_tmp, lmd_tmp, b_tmp;
//             lm_tmp = lm_cpy.block<3, 1>(i, 0);
//             lmd_tmp = lmd_cpy.block<3, 1>(i, 0);
//             b_tmp = b_cpy.block<3, 1>(i, 0);

//             // this.lambdad(i:i+2)' * this.J1I(i:i+2,:)
//             Eigen::Matrix<float, 3, 6> sJ1I = J1I_cpy.block<3, 6>(i, 0);
//             J1I_cgCpy.block<1, 6>(i, 0) = lmd_tmp.transpose() * sJ1I;
//             if (i + 1 < n) {
//                 J1I_cgCpy.block<2, 6>(i + 1, 0).setZero();
//             }

//             // this.lambdad(i:i+2)' * this.J2I(i:i+2,:)
//             Eigen::Matrix<float, 3, 6> sJ2I = J2I_cpy.block<3, 6>(i, 0);
//             J2I_cgCpy.block<1, 6>(i, 0) = lmd_tmp.transpose() * sJ2I;
//             if (i + 1 < n) {
//                 J1I_cgCpy.block<2, 6>(i + 1, 0).setZero();
//             }

//             // vec24f' * vec24 == 1x24 * 24x1 == 1x1 (dot product)
//             b_cgCpy(i) = lmd_tmp.dot(b_tmp);
//             if (i + 1 < n) {
//                 b_cgCpy(i + 1) = 0.0f;
//             }
//             if (i + 2 < n) {
//                 b_cgCpy(i + 2) = 0.0f;
//             }

//             // this.lambdad(i:i+2)' * this.lambda(i:i+2)
//             lm_cpy(i) = lmd_tmp.dot(lm_tmp);
//             if (i + 1 < n) {
//                 lm_cpy(i + 1) = 0.0f;
//             }
//             if (i + 2 < n) {
//                 lm_cpy(i + 2) = 0.0f;
//             }
//         }
//     }

//     clr.J1I_cg(J1I_cgCpy);
//     clr.J2I_cg(J2I_cgCpy);
//     clr.b_cg(b_cgCpy);
//     clr.lambda(lm_cpy);
//     clr.Minv_cg(vec24f::Ones());
// }

/*
function compute_degenerate_LTlambda(this)
    this.body1.LTx = this.body1.LTx + this.J1I_cg(this.freeIndex,:)' *
this.lambda(this.freeIndex); this.body2.LTx = this.body2.LTx +
this.J2I_cg(this.freeIndex,:)' * this.lambda(this.freeIndex); end
*/
inline void Collision::compute_degenerate_LTlambda(CollisionReference clr) {
    const auto& J2I_cgCpy = clr.J2I_cg();
    const auto& J1I_cgCpy = clr.J1I_cg();
    const auto& lm_cpy = clr.lambda();
    const auto& fidx_cpy = clr.freeIndex();

    auto b1 = clr.body1().get_rigid();
    auto b2 = clr.body2().get_rigid();
    auto b1_LTx = b1.LTx();
    auto b2_LTx = b2.LTx();

    int free_rows[24] = {0};
    unsigned int n = lm_cpy.size();
    unsigned int row_ct = 0;
    for (unsigned int i = 0; i < n; i++) {
        if (fidx_cpy(i)) {
            free_rows[row_ct++] = i;
        }
    }

    Eigen::Matrix<float, 24, 1> lm_free;
    for (unsigned int r = 0; r < row_ct; r++) {
        lm_free(r) = lm_cpy(free_rows[r]);
    }

    Eigen::Matrix<float, 24, 6> J1I_free;
    for (unsigned int r = 0; r < row_ct; r++) {
        int r_idx = free_rows[r];
        J1I_free.row(r) = J1I_cgCpy.row(r_idx);
    }

    b1_LTx += J1I_free.topRows(row_ct).transpose() * lm_free.topRows(row_ct);

    Eigen::Matrix<float, 24, 6> J2I_free;
    for (unsigned int r = 0; r < row_ct; r++) {
        int r_idx = free_rows[r];
        J2I_free.row(r) = J2I_cgCpy.row(r_idx);
    }

    b2_LTx += J2I_free.topRows(row_ct).transpose() * lm_free.topRows(row_ct);

    b1.LTx(b1_LTx);

    // TODO: Optimize (see Collision::compute_LTp)
    if (!this->is_ground(clr)) {
        b2.LTx(b2_LTx);
    }

    if (b1.LTx().hasNaN()) {
        TRACE("b1.LTx has NaN");
        exit(1);
    }
    if (b2.LTx().hasNaN()) {
        TRACE("b2.LTx has NaN");
        exit(1);
    }
}

/*
function compute_LTd_cg(this)
    this.body1.LTx = this.body1.LTx + this.J1I_cg(this.freeIndex,:)' *
this.d_cg(this.freeIndex); this.body2.LTx = this.body2.LTx +
this.J2I_cg(this.freeIndex,:)' * this.d_cg(this.freeIndex); end
*/
inline void Collision::compute_LTd_cg(CollisionReference clr) {
    const auto& J1I_cgCpy = clr.J1I_cg();
    const auto& J2I_cgCpy = clr.J2I_cg();
    const auto& d_cgCpy = clr.d_cg();
    const auto& fidx_cpy = clr.freeIndex();

    auto b1 = clr.body1().get_rigid();
    auto b2 = clr.body2().get_rigid();
    auto b1_LTx = b1.LTx();
    auto b2_LTx = b2.LTx();

    int free_rows[24];
    unsigned int row_ct = 0;
    unsigned int n = d_cgCpy.size();
    for (unsigned int i = 0; i < n; i++) {
        if (fidx_cpy(i) != 0) {
            free_rows[row_ct++] = i;
        }
    }

    Eigen::Matrix<float, 24, 1> d_free;
    for (unsigned int r = 0; r < row_ct; r++) {
        d_free(r) = d_cgCpy(free_rows[r]);
    }

    Eigen::Matrix<float, 24, 6> J1I_free;
    for (unsigned int r = 0; r < row_ct; r++) {
        int idx = free_rows[r];
        J1I_free.row(r) = J1I_cgCpy.row(idx);
    }

    b1_LTx += J1I_free.topRows(row_ct).transpose() * d_free.topRows(row_ct);

    Eigen::Matrix<float, 24, 6> J2I_free;
    for (unsigned int r = 0; r < row_ct; r++) {
        int idx = free_rows[r];
        J2I_free.row(r) = J2I_cgCpy.row(idx);
    }
    b2_LTx += J2I_free.topRows(row_ct).transpose() * d_free.topRows(row_ct);

    b1.LTx(b1_LTx);

    // TODO: Optimize (see Collision::compute_LTp)
    if (!this->is_ground(clr)) {
        b2.LTx(b2_LTx);
    }
}

/*
function compute_LTp(this)
    this.body1.LTx = this.body1.LTx + this.J1I' * this.p;
    this.body2.LTx = this.body2.LTx + this.J2I' * this.p;
end
*/
inline void Collision::compute_LTp(CollisionReference clr) {
    const auto& J1I_cpy = clr.J1I();
    const auto& J2I_cpy = clr.J2I();
    const auto& p_cpy = clr.p();

    auto b1 = clr.body1().get_rigid();
    auto b2 = clr.body2().get_rigid();
    auto b1_LTx = b1.LTx();
    auto b2_LTx = b2.LTx();

    b1_LTx += J1I_cpy.transpose() * p_cpy;
    b2_LTx += J2I_cpy.transpose() * p_cpy;

    /*
        If we have a ground collision, MATLAB still updates both bodies,
       although only body1 should represent our ground plane. However body2() is
       still initialized with index 0, which overwrites body1 because of the
       global SOA logic. In fact much of the MATLAB functions unnecessarily
       perform operations and  overwrite buffers that are unused in the ground
        case, but it is never penalized because of AOS

        Thus, we should add our own checks for this case.
        TODO: Optimize by two separate if else, one which doesn't do anything
       with J2I / b2_LTX
    */
    b1.LTx(b1_LTx);
    if (!this->is_ground(clr)) {
        b2.LTx(b2_LTx);
    }
}

/*
function compute_LLTx(this)
    this.Ax = this.J1I * this.body1.LTx + this.J2I * this.body2.LTx;
end
*/
inline void Collision::compute_LLTx(CollisionReference clr) {
    const auto& J1I_cpy = clr.J1I();
    const auto& J2I_cpy = clr.J2I();
    const auto& b1 = clr.body1().get_rigid();
    const auto& b2 = clr.body2().get_rigid();
    const auto& LTx1 = b1.LTx();
    const auto& LTx2 = b2.LTx();

    clr.Ax(J1I_cpy * LTx1 + J2I_cpy * LTx2);
}

/*
function compute_degenerate_LLTx(this)
    this.Ax = this.J1I_cg * this.body1.LTx + this.J2I_cg * this.body2.LTx;
end
*/
inline void Collision::compute_degenerate_LLTx(CollisionReference clr) {
    const auto& J1I_cgCpy = clr.J1I_cg();
    const auto& J2I_cgCpy = clr.J2I_cg();
    const auto& b1 = clr.body1().get_rigid();
    const auto& b2 = clr.body2().get_rigid();
    const auto& LTx1 = b1.LTx();
    const auto& LTx2 = b2.LTx();

    clr.Ax(J1I_cgCpy * LTx1 + J2I_cgCpy * LTx2);
}

/*
function tbar_i = compute_tbar(this)
    for i = 1:3:3*this.contactNum
        t =
Collision.rayConeIntersection(this.lambda(i:i+2),-this.g(i:i+2),this.mu);
        this.t_bar(i) = t;
        if((abs(this.g(i)) < 1e-9 && abs(this.lambda(i))<1e-9))
            this.t_bar(i+1) = 0;
        elseif(this.g(i)>0)
            this.t_bar(i+1) = this.lambda(i) / this.g(i);
        else
            this.t_bar(i+1) = Inf;
        end
        this.t_bar(i+2) = Inf;
    end

    tbar_i = this.t_bar;
end
*/
/*

FIXME: Getting a zeroed out g_seg right now, which originates from an erroneous Ax update caused from a zeroed LTx
        which is caused by zeroed lambda in compute_LTlambda 

*/
inline vec24f Collision::compute_tbar(CollisionReference clr) {
    const float POSINF = 1e20;
    const float EPS = 1e-9f;
    const auto& lm_cpy = clr.lambda();
    const auto& g_cpy = clr.g();
    auto tbar_cpy = clr.t_bar();
    float mu_cpy = clr.mu();
    unsigned int n = 3 * clr.contactNum();

    for (unsigned int i = 0; i < n; i += 3) {
        Eigen::Vector3f lm_seg = lm_cpy.block<3, 1>(i, 0);
        Eigen::Vector3f g_seg = g_cpy.block<3, 1>(i, 0);
        if (g_seg.hasNaN()) {
            printf("NaN detected in g_seg at contact i=%u.\n", i);
            exit(1);
        }
        float t = this->rayConeIntersection(lm_seg, -g_seg, mu_cpy);
        tbar_cpy(i) = t;

        if (fabsf(g_cpy(i)) < EPS && fabsf(lm_cpy(i)) < EPS) {
            tbar_cpy(i + 1) = 0.0f;
        } else if (g_cpy(i) > 0.0f) {
            tbar_cpy(i + 1) = lm_cpy(i) / g_cpy(i);
        } else {
            tbar_cpy(i + 1) = POSINF;
        }
        tbar_cpy(i + 2) = POSINF;
    }

    clr.t_bar(tbar_cpy);

    return tbar_cpy;
}

/*
function compute_lambdac(this, t)
    for i = 1:3:3*this.contactNum
        if(t < this.t_bar(i))
            this.lambdac(i:i+2) = this.lambda(i:i+2) - t*this.g(i:i+2);
        elseif(t < this.t_bar(i+1))
            gp = this.g(i:i+2);
            if(norm(this.lambda(i:i+2))<1e-9)
                this.lambdad(i:i+2) = [1 0 0]';
            else
                this.lambdad(i:i+2) = this.lambda(i:i+2) /
norm(this.lambda(i:i+2)); end gp =
(this.lambdad(i:i+2)'*gp)*this.lambdad(i:i+2); this.lambdac(i:i+2) =
this.lambda(i:i+2) - this.t_bar(i)*this.g(i:i+2) - (t - this.t_bar(i))*gp; else
            gp = this.g(i:i+2);
            if(norm(this.lambda(i:i+2))<1e-9)
                this.lambdad(i:i+2) = [1 0 0]';
            else
                this.lambdad(i:i+2) = this.lambda(i:i+2) /
norm(this.lambda(i:i+2)); end gp =
(this.lambdad(i:i+2)'*gp)*this.lambdad(i:i+2); this.lambdac(i:i+2) =
this.lambda(i:i+2) -this.t_bar(i)*this.g(i:i+2) + (this.t_bar(i+1) -
this.t_bar(i))*gp; end end end
*/
inline void Collision::compute_lambdac(CollisionReference clr, float t) {
    auto lm_cpy = clr.lambda();
    auto lmc_cpy = clr.lambdac();
    auto lmd_cpy = clr.lambdad();
    auto g_cpy = clr.g();
    auto tb_cpy = clr.t_bar();

    unsigned int n = 3 * clr.contactNum();
    for (unsigned int i = 0; i < n; i += 3) {
        float t0(tb_cpy(i)), t1(tb_cpy(i + 1));

        Eigen::Vector3f lm_seg, g_seg;
        lm_seg = lm_cpy.block<3, 1>(i, 0);
        g_seg = g_cpy.block<3, 1>(i, 0);

        // if(t < this.t_bar(i))
        if (t < t0) {
            lmc_cpy.block<3, 1>(i, 0) = lm_seg - t * g_seg;
        } else if (t < t1) {
            Eigen::Vector3f gp = g_seg;
            float lm_segNorm = lm_seg.norm();

            if (lm_segNorm < 1e-9f) {
                lmd_cpy.block<3, 1>(i, 0) = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
            } else {
                lmd_cpy.block<3, 1>(i, 0) = lm_seg / lm_segNorm;
            }

            gp = (lmd_cpy.block<3, 1>(i, 0).transpose() * gp) *
                 lmd_cpy.block<3, 1>(i, 0);
            lmc_cpy.block<3, 1>(i, 0) = lm_seg - t0 * g_seg - (t - t0) * gp;
        } else {
            Eigen::Vector3f gp = g_seg;
            if (lm_seg.norm() < 1e-9f) {
                lmd_cpy.block<3, 1>(i, 0) = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
            } else {
                lmd_cpy.block<3, 1>(i, 0) = lm_seg / lm_seg.norm();
            }

            gp = (lmd_cpy.block<3, 1>(i, 0).transpose() * gp) *
                 lmd_cpy.block<3, 1>(i, 0);
            lmc_cpy.block<3, 1>(i, 0) = lm_seg - t0 * g_seg + (t1 - t0) * gp;
        }
    }

    clr.lambdac(lmc_cpy);
    clr.lambdad(lmd_cpy);
}

/*
function compute_lambdad(this, t)
    for i = 1:3:3*this.contactNum
        if(t<this.t_bar(i))
            this.freeIndex(i:i+2) = true;
        elseif(t<this.t_bar(i+1))
            this.freeIndex(i) = true;
            this.freeIndex(i+1:i+2) = false;
        else
            this.freeIndex(i:i+2) = false;
        end

        if(norm(this.lambdac(i:i+2))<1e-9)
            this.lambdad(i:i+2) = [1 0 0]';
        else
            this.lambdad(i:i+2) = this.lambdac(i:i+2) /
norm(this.lambdac(i:i+2)); end end end
*/
inline void Collision::compute_lambdad(CollisionReference clr, float t) {
    const auto& lmc_cpy = clr.lambdac();
    auto lmd_cpy = clr.lambdad();
    auto fidx_cpy = clr.freeIndex();
    const auto& tb_cpy = clr.t_bar();

    unsigned int n = 3 * clr.contactNum();
    for (unsigned int i = 0; i < n; i += 3) {
        float t0(tb_cpy(i)), t1(tb_cpy(i + 1));

        // if(t<this.t_bar(i))
        if (t < t0) {
            fidx_cpy.block<3, 1>(i, 0).setOnes();
        } else if (t < t1) {
            fidx_cpy(i) = 1;
            fidx_cpy.block<2, 1>(i + 1, 0).setZero();
        } else {
            fidx_cpy.block<3, 1>(i, 0).setZero();
        }

        Eigen::Vector3f lmc_seg = lmc_cpy.block<3, 1>(i, 0);
        float lmc_segNorm = lmc_seg.norm();
        if (lmc_segNorm < 1e-9f) {
            lmd_cpy.block<3, 1>(i, 0) = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
        } else {
            lmd_cpy.block<3, 1>(i, 0) = lmc_seg / lmc_segNorm;
        }
    }

    clr.lambdad(lmd_cpy);
    clr.freeIndex(fidx_cpy);
}

/*
function compute_p(this,t)
    for i = 1:3:3*this.contactNum
        if(t < this.t_bar(i))
            this.p(i:i+2) = -this.g(i:i+2);
        elseif(t < this.t_bar(i+1))
            gp = this.g(i:i+2);
            if(norm(this.lambda(i:i+2))<1e-9)
                this.lambdad(i:i+2) = [1 0 0]';
            else
                this.lambdad(i:i+2) = this.lambda(i:i+2) /
norm(this.lambda(i:i+2)); end gp =
(this.lambdad(i:i+2)'*gp)*this.lambdad(i:i+2); this.p(i:i+2) = -gp; else
            this.p(i:i+2) = zeros(3,1);
        end
    end
end
*/
inline void Collision::compute_p(CollisionReference clr, float t) {
    auto lm_cpy = clr.lambda();
    auto lmd_cpy = clr.lambdad();
    const auto& g_cpy = clr.g();
    auto p_cpy = clr.p();
    const auto& tb_cpy = clr.t_bar();

    unsigned int n = 3 * clr.contactNum();
    for (unsigned int i = 0; i < n; i += 3) {
        float t0(tb_cpy(i)), t1(tb_cpy(i + 1));

        Eigen::Vector3f g_seg = g_cpy.block<3, 1>(i, 0);
        if (t < t0) {
            p_cpy.block<3, 1>(i, 0) = -g_seg;
        } else if (t < t1) {
            Eigen::Vector3f gp = g_seg;
            float lm_segNorm = lm_cpy.block<3, 1>(i, 0).norm();

            if (lm_segNorm < 1e-9f) {
                lmd_cpy.block<3, 1>(i, 0) = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
            } else {
                lmd_cpy.block<3, 1>(i, 0) =
                    lm_cpy.block<3, 1>(i, 0) / lm_segNorm;
            }

            gp = (lmd_cpy.block<3, 1>(i, 0).transpose() * gp) *
                 lmd_cpy.block<3, 1>(i, 0);
            p_cpy.block<3, 1>(i, 0) = -gp;
        } else {
            p_cpy.block<3, 1>(i, 0).setZero();
        }
    }

    clr.p(p_cpy);
    clr.lambdad(lmd_cpy);
}

/*
function project(this)
    for i = 1:3:3*this.contactNum
        if(this.lambda(i) < 0)
            this.lambda(i) = 0;
        end
        if(this.freeIndex(i) && ~this.freeIndex(i+1))
            this.lambda(i:i+2) = this.lambda(i) * this.lambdad(i:i+2);
        else
            if (norm(this.lambda(i+1:i+2)) > this.mu * this.lambda(i))
                scale = this.mu * this.lambda(i) / norm(this.lambda(i+1:i+2));
                this.lambda(i+1:i+2) = scale * this.lambda(i+1:i+2);
            end
        end
    end
end
*/
inline void Collision::project(CollisionReference clr) {
    auto lm_cpy = clr.lambda();
    const auto& lmd_cpy = clr.lambdad();
    const auto& fidx_cpy = clr.freeIndex();
    float mu_cpy = clr.mu();

    unsigned int n = 3 * clr.contactNum();
    for (unsigned int i = 0; i < n; i += 3) {
        if (lm_cpy(i) < 0) {
            lm_cpy(i) = 0;
        }

        if (fidx_cpy(i) && !fidx_cpy(i + 1)) {
            lm_cpy.block<3, 1>(i, 0) = lm_cpy(i) * lmd_cpy.block<3, 1>(i, 0);
        } else {
            float lm_norm = lm_cpy.block<2, 1>(i + 1, 0).norm();
            if (lm_norm > mu_cpy * lm_cpy(i)) {
                float scale = mu_cpy * lm_cpy(i) / lm_norm;
                lm_cpy.block<2, 1>(i + 1, 0) =
                    scale * lm_cpy.block<2, 1>(i + 1, 0);
            }
        }
    }

    clr.lambda(lm_cpy);
}

/*
function feasible = update_cg(this, alpha)
    feasible = true;
    this.lambda(this.freeIndex) = this.lambda(this.freeIndex) + alpha *
this.d_cg(this.freeIndex); for i = 1:3:3*this.contactNum if(this.freeIndex(i) &&
this.lambda(i) < 0) feasible = false; end

        if (this.freeIndex(i+1) && norm(this.lambda(i+1:i+2)) > this.mu *
this.lambda(i)) feasible = false; end end end
*/
inline bool Collision::update_cg(CollisionReference clr, float alpha) {
    bool feasible = true;
    auto lam_cpy = clr.lambda();
    const auto& d_cgCpy = clr.d_cg();
    const auto& fidx_cpy = clr.freeIndex();
    float mu = clr.mu();
    unsigned int n = 3 * clr.contactNum();

    for (unsigned int i = 0; i < n; i++) {
        if (fidx_cpy(i) != 0) {
            lam_cpy(i) += alpha * d_cgCpy(i);
        }
    }

    for (unsigned int i = 0; i < n; i += 3) {
        if (fidx_cpy(i) != 0 && lam_cpy(i) < 0.0f) {
            feasible = false;
        }

        if ((i + 1 < n) && (fidx_cpy(i + 1) != 0)) {
            float tangMag = std::sqrt(lam_cpy(i + 1) * lam_cpy(i + 1) +
                                      lam_cpy(i + 2) * lam_cpy(i + 2));
            if (tangMag > mu * lam_cpy(i)) {
                feasible = false;
            }
        }
    }

    clr.lambda(lam_cpy);
    return feasible;
}

/*
%% Ray-Cone Intersection
function t = rayConeIntersection(x, g, mu)
    gnorm = norm(g);
    g = g / gnorm;

    A = g(2)^2 + g(3)^2 - g(1)^2 * mu^2;
    B = 2 * (x(2)*g(2) + x(3)*g(3) - x(1)*g(1) * mu^2);
    C = x(2)^2 + x(3)^2 - x(1)^2 * mu^2;

    if(abs(C)<1e-9)
        C = 0;
    end

    if(abs(C)<1e-9 && gnorm < 1e-9)
        t = 0;
        return;
    end

    % Solve the quadratic equation
    discriminant = B^2 - 4*A*C;

    if(abs(A)<1e-12)
        if(g(1)>0)
            t = Inf;
            return;
        end

        if(abs(B)<1e-12)
            t = - x(1) / g(1);
        else
            t = -C/B;
        end
        t = t * gnorm;
        return;
    end

    if discriminant < 1e-12
        if g(1)>0
            if(norm(g(2:3)) > mu*g(1))
                t = 0;
            else
                t = Inf;
            end
        else
            t = -B / (2 * A);
        end
        t = t * gnorm;
        return;
    end

    % Compute the roots
    if(A>0)
        t1 = (-B - sqrt(discriminant)) / (2 * A);
        t2 = (-B + sqrt(discriminant)) / (2 * A);
    else
        t2 = (-B - sqrt(discriminant)) / (2 * A);
        t1 = (-B + sqrt(discriminant)) / (2 * A);
    end

    if(t1 ==0)
        if(x(1) + t2*g(1) >0)
            t = t2;
        else
            t = t1;
        end
        t = t * gnorm;
        return;
    end

    if(t2 ==0)
        if(x(1) + t1*g(1) >0)
            t = t2;
        else
            t = Inf;
        end
        t = t * gnorm;
        return;
    end


    if(t1>0 && t2 >0)
        t = min(t1,t2);
    elseif(t1<0 && t2<0)
        t = Inf;
    else
        t = max(t1,t2);
    end
    t = t * gnorm;
end
*/
inline float Collision::rayConeIntersection(const Eigen::Vector3f& x,
                                            const Eigen::Vector3f& g,
                                            float mu) {
    const float EPS1(1e-9f), EPS2(1e-12f), POSINF(1e20);
    float gnorm = g.norm();
    Eigen::Vector3f g_norm = g / gnorm;

    if (g_norm.hasNaN()) {
        TRACE("g_norm NaN detected!");
        exit(1);
    }

    float mu_sqr = mu * mu;
    float A = g_norm(1) * g_norm(1) + g_norm(2) * g_norm(2) -
              g_norm(0) * g_norm(0) * mu_sqr;
    float B =
        2 * (x(1) * g_norm(1) + x(2) * g_norm(2) - x(0) * g_norm(0) * mu_sqr);
    float C = x(1) * x(1) + x(2) * x(2) - x(0) * x(0) * mu_sqr;

    if (fabsf(C) < EPS1) {
        C = 0.0f;
    }

    if (fabsf(C) < EPS1 && gnorm < EPS2) {
        return 0.0f;
    }

    float t;
    float discriminant = B * B - 4 * A * C;
    if (fabsf(A) < EPS2) {
        if (g_norm(0) > 0.0f) {
            return POSINF;
        }

        if (fabsf(B) < EPS2) {
            t = -x(0) / g_norm(0);
        } else {
            t = -C / B;
        }

        return t * gnorm;
    }

    if (discriminant < EPS2) {
        if (g_norm(0) > 0.0f) {
            if (g_norm.tail<2>().norm() > mu * g_norm(0)) {
                t = 0.0f;
            } else {
                t = POSINF;
            }
        }
        else {
            t = -B / (2 * A);
        }

        return t * gnorm;
    }

    float t1, t2;
    if (A > 0.0f) {
        t1 = (-B - sqrt(discriminant)) / (2 * A);
        t2 = (-B + sqrt(discriminant)) / (2 * A);
    }
    else {
        t2 = (-B - sqrt(discriminant)) / (2 * A);
        t1 = (-B + sqrt(discriminant)) / (2 * A);
    }

    if (t1 == 0.0f) {
        if (x(0) + t2 * g_norm(0) > 0.0f) {
            t = t2;
        } else {
            t = t1;
        }

        return t * gnorm;
    }

    if (t2 == 0.0f) {
        if (x(0) + t1 * g_norm(0) > 0.0f) {
            t = t2;
        } else {
            t = POSINF;
        }

        return t * gnorm;
    }

    if (t1 > 0.0f && t2 > 0.0f) {
        t = std::min(t1, t2);
    } else if (t1 < 0.0f && t2 < 0.0f) {
        t = POSINF;
    } else {
        t = std::max(t1, t2);
    }

    return t * gnorm;
}

}  // namespace apbd
