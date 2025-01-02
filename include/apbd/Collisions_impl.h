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

/*
        function computeJ_b(this)
            if(this.ground)
                %I1sqrt = 1 ./ sqrt([this.body1.Mr; ones(3,1)*this.body1.Mp]);
                for i = 1:this.contactNum
                    rows = 3*(i-1) + 1: 3*i;
                    this.J1I(rows,1:3) = this.constraints{i}.raXnI';
                    this.J1I(rows,4:6) = this.constraints{i}.contactFrame' ./ sqrt(this.body1.Mp);
                    this.b(rows) = -this.constraints{i}.evalCs();
                end
            else
                for i = 1:this.contactNum
                    rows = 3*(i-1) + 1: 3*i;
                    this.J1I(rows,1:3) = this.constraints{i}.raXnI1';
                    this.J1I(rows,4:6) = this.constraints{i}.contactFrame' ./ sqrt(this.constraints{i}.body1.Mp);

                    this.J2I(rows,1:3) = -this.constraints{i}.raXnI2';
                    this.J2I(rows,4:6) = -this.constraints{i}.contactFrame' ./ sqrt(this.constraints{i}.body2.Mp);
                    this.b(rows) = -this.constraints{i}.evalCs();
                end
            end
        end
*/
inline void Collision::computeJ_b(CollisionReference clr, float h) {
    if (this->is_ground(clr)) {
        for (unsigned int i = 0; i < clr.contactNum(); i++) {
            unsigned int rows = 3 * i;
            clr.J1I().block<3, 3>(rows, 0) = this->constraints[i].get_rigid().raXnI1().transpose();
            clr.J1I().block<3, 3>(rows, 3) = this->constraints[i].get_rigid().contactFrame().transpose() / sqrt(clr.body1().get_rigid().Mp());
            float Cs = this->constraints[i].get_ground().evalCs(h);
            clr.b().block<3, 1>(rows, 0) = Eigen::Vector3f::Constant(-Cs);
        }
    }
    else {
        for (unsigned int i = 0; i < clr.contactNum(); i++) {
            unsigned int rows = 3 * i;
            auto raXnI1 = this->constraints[i].get_rigid().raXnI1();
            auto raXnI2 = this->constraints[i].get_rigid().raXnI2();
            auto c_frame = this->constraints[i].get_rigid().contactFrame();
            auto b1_Mp = clr.body1().get_rigid().Mp();
            auto b2_Mp = clr.body2().get_rigid().Mp();

            clr.J1I().block<3, 3>(rows, 0) = raXnI1.transpose();
            clr.J1I().block<3, 3>(rows, 3) = c_frame.transpose() / sqrt(b1_Mp);

            clr.J2I().block<3, 3>(rows, 0) = -raXnI2.transpose();
            clr.J2I().block<3, 3>(rows, 3) = -c_frame.transpose() / sqrt(b2_Mp);

            float Cs = this->constraints[i].get_rigid().evalCs(h);
            clr.b().block<3, 1>(rows, 0) = Eigen::Vector3f::Constant(-Cs);
        }
    }
}

/*
        function compute_LTlambda(this)
            this.body1.LTx = this.body1.LTx + this.J1I' * this.lambda;
            this.body2.LTx = this.body2.LTx + this.J2I' * this.lambda;
        end

        %%
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

        %%
        function compute_degenerate_LTlambda(this)
            this.body1.LTx = this.body1.LTx + this.J1I_cg(this.freeIndex,:)' * this.lambda(this.freeIndex);
            this.body2.LTx = this.body2.LTx + this.J2I_cg(this.freeIndex,:)' * this.lambda(this.freeIndex);
        end

        %%
        function compute_LTd_cg(this)
            this.body1.LTx = this.body1.LTx + this.J1I_cg(this.freeIndex,:)' * this.d_cg(this.freeIndex);
            this.body2.LTx = this.body2.LTx + this.J2I_cg(this.freeIndex,:)' * this.d_cg(this.freeIndex);
        end

        %%
        function compute_LTp(this)
            this.body1.LTx = this.body1.LTx + this.J1I' * this.p;
            this.body2.LTx = this.body2.LTx + this.J2I' * this.p;
        end

        %%
        function compute_LLTx(this)
            this.Ax = this.J1I * this.body1.LTx + this.J2I * this.body2.LTx;
        end

        %%
        function compute_degenerate_LLTx(this)
            this.Ax = this.J1I_cg * this.body1.LTx + this.J2I_cg * this.body2.LTx;
        end

        %%
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

        %%
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

        %%
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

        %%
        function compute_p(this,t)
            for i = 1:3:3*this.contactNum
                if(t < this.t_bar(i))
                    this.p(i:i+2) = -this.g(i:i+2);
                elseif(t < this.t_bar(i+1))
                    gp = this.g(i:i+2);
                    if(norm(this.lambda(i:i+2))<1e-9)
                        this.lambdad(i:i+2) = [1 0 0]';
                    else
                        this.lambdad(i:i+2) = this.lambda(i:i+2) / norm(this.lambda(i:i+2));
                    end
                    gp = (this.lambdad(i:i+2)'*gp)*this.lambdad(i:i+2);
                    this.p(i:i+2) = -gp;
                else
                    this.p(i:i+2) = zeros(3,1);
                end
            end
        end

        %%
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

        %%
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
    end
*/

inline void Collision::compute_LTlambda(CollisionReference clr) {
    const auto& body1_LTx_0 = clr.body1().get_rigid().LTx();
    const auto& body2_LTx_0 = clr.body1().get_rigid().LTx();

    // clr.body1().get_rigid().LTx(body1_LTx_0 + clr.J1I() * clr.lambda());
    // clr.body1().get_rigid().LTx(body2_LTx_0 + clr.J1I() * clr.lambda());
}

inline void Collision::compute_degenerate_J1I_J2I_b(CollisionReference clr) {
    clr.J1I_cg(clr.J1I());
    clr.J2I_cg(clr.J2I());
    clr.b_cg(clr.b());
    unsigned int n = clr.contactNum();
    for (unsigned int i = 0; i < 3 * n; i += 3) {
        if (clr.freeIndex()(i) && (i + 1 < n) && !clr.freeIndex()(i + 1)) {
            clr.J1I_cg().block<1, 6>(i, 0) = clr.lambdad().segment(i, 3).transpose() * clr.J1I().block<3, 6>(i, 0);
            clr.J2I_cg().block<1, 6>(i, 0) = clr.lambdad().segment(i, 3).transpose() * clr.J2I().block<3, 6>(i, 0);
            clr.b_cg()(i) = clr.lambdad().segment(i, 3).transpose() * clr.b().segment(i, 3);
            clr.lambda()(i) = clr.lambdad().segment(i, 3).transpose() * clr.lambda().segment(i, 3);
        }
    }

}

inline void Collision::compute_degenerate_LTlambda(CollisionReference clr) {
    const auto& body1_LTx_0 = clr.body1().get_rigid().LTx();
    const auto& body2_LTx_0 = clr.body1().get_rigid().LTx();
    /* TODO: */
}

inline void Collision::compute_LTd_cg(CollisionReference clr) {
    const auto& body1_LTx_0 = clr.body1().get_rigid().LTx();
    const auto& body2_LTx_0 = clr.body1().get_rigid().LTx();
    /* TODO: */
}

inline void Collision::compute_LTp(CollisionReference clr) {
    const auto& body1_LTx_0 = clr.body1().get_rigid().LTx();
    const auto& body2_LTx_0 = clr.body1().get_rigid().LTx();
    /* TODO: */
}

inline void Collision::compute_LLTx(CollisionReference clr) {
    const auto& body1_LTx_0 = clr.body1().get_rigid().LTx();
    const auto& body2_LTx_0 = clr.body1().get_rigid().LTx();
    /* TODO: */
}

inline void Collision::compute_degenerate_LLTx(CollisionReference clr) {
    const auto& body1_LTx_0 = clr.body1().get_rigid().LTx();
    const auto& body2_LTx_0 = clr.body1().get_rigid().LTx();
    /* TODO: */
    clr.Ax(clr.J1I_cg() * body1_LTx_0 + clr.J2I_cg() * body2_LTx_0);
}

// TODO: fix return type
inline void Collision::compute_tbar(CollisionReference clr) {
    /* TODO */
}

inline void Collision::compute_lambdac(CollisionReference clr, float t) {
    /* TODO */
}

inline void Collision::compute_lambdad(CollisionReference clr, float t) {
    /* TODO */
}

inline void Collision::compute_p(CollisionReference clr, float t) {
    /* TODO */
}

inline void Collision::project(CollisionReference clr) {
    /* TODO */
}

inline bool Collision::update_cg(CollisionReference clr, float alpha) {
    /* TODO */
}

}  // namespace apbd
