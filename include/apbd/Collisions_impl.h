#pragma once
#include "apbd/BodyReference_impl.h"
#include "apbd/Collisions.h"
#include "apbd/ConstraintReference_impl.h"

namespace apbd {

inline void Collision::setContacts(
    CollisionReference clr,
    const cuda::std::pair<cuda::std::array<Contact, 8>, size_t> cdata) {
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
                                      size_t &ground_count,
                                      size_t &rigid_count) {
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

// TODO: I don't like the structure of being forced to return a copy when updating
inline void Collision::computeJ_b(CollisionReference clr, float h) {
    unsigned int n = clr.contactNum();

    if (this->is_ground(clr)) {
        for (unsigned int i = 0; i < n; i++) {
            unsigned int rows = 3 * i;
            auto grd = this->constraints[i].get_ground();

            Eigen::Matrix3f raXnI = grd.raXnI1();
            Eigen::Matrix3f c_frame = grd.contactFrame();
            // TODO: Is it clr.body1() or grd.body() ?
            float b1_Mp = clr.body1().get_rigid().Mp();

            auto J1I_cpy = clr.J1I();
            auto b_cpy = clr.b();

            J1I_cpy.block<3, 3>(rows, 0) = raXnI.transpose();
            J1I_cpy.block<3, 3>(rows, 3) = c_frame.transpose() / sqrt(b1_Mp);
            b_cpy.block<3, 1>(rows, 0) = -grd.evalCs(h);

            clr.J1I(J1I_cpy);
            clr.b(b_cpy);
        }
    }
    else {
        for (unsigned int i = 0; i < n; i++) {
            unsigned int rows = 3 * i;
            
            auto rig = this->constraints[i].get_rigid();

            Eigen::Matrix3f raXnI1 = rig.raXnI1();
            Eigen::Matrix3f c_frame1 = rig.contactFrame();
            float b1_Mp = rig.body1().Mp();
            
            Eigen::Matrix3f raXnI2 = rig.raXnI2();
            Eigen::Matrix3f c_frame2 = rig.contactFrame();
            float b2_Mp = rig.body2().Mp();

            auto J1I_cpy = clr.J1I();
            auto J2I_cpy = clr.J2I();
            auto b_cpy = clr.b();

            J1I_cpy.block<3, 3>(rows, 0) = raXnI1.transpose();
            J1I_cpy.block<3, 3>(rows, 3) = c_frame1.transpose() / sqrt(b1_Mp);

            J2I_cpy.block<3, 3>(rows, 0) = -raXnI2.transpose();
            J2I_cpy.block<3, 3>(rows, 3) = -c_frame2.transpose() / sqrt(b2_Mp);

            b_cpy.block<3, 1>(rows, 0) = -rig.evalCs(h);

            clr.J1I(J1I_cpy);
            clr.J2I(J2I_cpy);
            clr.b(b_cpy);
        }
    }
}

/*
function compute_LTlambda(this)
    this.body1.LTx = this.body1.LTx + this.J1I' * this.lambda;
    this.body2.LTx = this.body2.LTx + this.J2I' * this.lambda;
end
*/
inline void Collision::compute_LTlambda(CollisionReference clr) {
    const auto& body1_LTx_0 = clr.body1().get_rigid().LTx();
    const auto& body2_LTx_0 = clr.body1().get_rigid().LTx();
    
    clr.body1().get_rigid().LTx(body1_LTx_0 + clr.J1I().transpose() * clr.lambda());
    clr.body2().get_rigid().LTx(body2_LTx_0 + clr.J2I().transpose() * clr.lambda());
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
inline void Collision::compute_degenerate_J1I_J2I_b(CollisionReference clr) {

}

/*
function compute_degenerate_LTlambda(this)
    this.body1.LTx = this.body1.LTx + this.J1I_cg(this.freeIndex,:)' * this.lambda(this.freeIndex);
    this.body2.LTx = this.body2.LTx + this.J2I_cg(this.freeIndex,:)' * this.lambda(this.freeIndex);
end
*/
inline void Collision::compute_degenerate_LTlambda(CollisionReference clr) {
    const auto& body1_LTx_0 = clr.body1().get_rigid().LTx();
    const auto& body2_LTx_0 = clr.body1().get_rigid().LTx();
    /* TODO: */
}

/*
function compute_LTd_cg(this)
    this.body1.LTx = this.body1.LTx + this.J1I_cg(this.freeIndex,:)' * this.d_cg(this.freeIndex);
    this.body2.LTx = this.body2.LTx + this.J2I_cg(this.freeIndex,:)' * this.d_cg(this.freeIndex);
end
*/
inline void Collision::compute_LTd_cg(CollisionReference clr) {
    const auto& body1_LTx_0 = clr.body1().get_rigid().LTx();
    const auto& body2_LTx_0 = clr.body1().get_rigid().LTx();
    /* TODO: */
}

/*
function compute_LTp(this)
    this.body1.LTx = this.body1.LTx + this.J1I' * this.p;
    this.body2.LTx = this.body2.LTx + this.J2I' * this.p;
end
*/
inline void Collision::compute_LTp(CollisionReference clr) {
    const auto& body1_LTx_0 = clr.body1().get_rigid().LTx();
    const auto& body2_LTx_0 = clr.body1().get_rigid().LTx();
    /* TODO: */
}

/*
function compute_LLTx(this)
    this.Ax = this.J1I * this.body1.LTx + this.J2I * this.body2.LTx;
end
*/
inline void Collision::compute_LLTx(CollisionReference clr) {
    const auto& body1_LTx_0 = clr.body1().get_rigid().LTx();
    const auto& body2_LTx_0 = clr.body1().get_rigid().LTx();
    /* TODO: */
}

/*
function compute_degenerate_LLTx(this)
    this.Ax = this.J1I_cg * this.body1.LTx + this.J2I_cg * this.body2.LTx;
end
*/
inline void Collision::compute_degenerate_LLTx(CollisionReference clr) {
    const auto& body1_LTx_0 = clr.body1().get_rigid().LTx();
    const auto& body2_LTx_0 = clr.body1().get_rigid().LTx();
    /* TODO: */
    clr.Ax(clr.J1I_cg() * body1_LTx_0 + clr.J2I_cg() * body2_LTx_0);
}

/*
function tbar_i = compute_tbar(this)
    for i = 1:3:3*this.contactNum
        t = Collision.rayConeIntersection(this.lambda(i:i+2),-this.g(i:i+2),this.mu);
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

    TODO: fix return type
*/
inline void Collision::compute_tbar(CollisionReference clr) {
    /* TODO */
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
                this.lambdad(i:i+2) = this.lambda(i:i+2) / norm(this.lambda(i:i+2));
            end
            gp = (this.lambdad(i:i+2)'*gp)*this.lambdad(i:i+2);
            this.lambdac(i:i+2) = this.lambda(i:i+2) - this.t_bar(i)*this.g(i:i+2) - (t - this.t_bar(i))*gp;
        else
            gp = this.g(i:i+2);
            if(norm(this.lambda(i:i+2))<1e-9)
                this.lambdad(i:i+2) = [1 0 0]';
            else
                this.lambdad(i:i+2) = this.lambda(i:i+2) / norm(this.lambda(i:i+2));
            end
            gp = (this.lambdad(i:i+2)'*gp)*this.lambdad(i:i+2);
            this.lambdac(i:i+2) = this.lambda(i:i+2) -this.t_bar(i)*this.g(i:i+2) + (this.t_bar(i+1) - this.t_bar(i))*gp;
        end
    end
end
*/
inline void Collision::compute_lambdac(CollisionReference clr, float t) {
    /* TODO */
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
                    this.lambdad(i:i+2) = this.lambdac(i:i+2) / norm(this.lambdac(i:i+2));
                end
            end
        end
*/
inline void Collision::compute_lambdad(CollisionReference clr, float t) {
    /* TODO */
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
inline void Collision::compute_p(CollisionReference clr, float t) {
    /* TODO */
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
    /* TODO */
}

/*
function feasible = update_cg(this, alpha)
    feasible = true;
    this.lambda(this.freeIndex) = this.lambda(this.freeIndex) + alpha * this.d_cg(this.freeIndex);
    for i = 1:3:3*this.contactNum
        if(this.freeIndex(i) && this.lambda(i) < 0)
            feasible = false;
        end

        if (this.freeIndex(i+1) && norm(this.lambda(i+1:i+2)) > this.mu * this.lambda(i))
            feasible = false;
        end
    end
end
*/
inline bool Collision::update_cg(CollisionReference clr, float alpha) {
    /* TODO */
}

}  // namespace apbd
