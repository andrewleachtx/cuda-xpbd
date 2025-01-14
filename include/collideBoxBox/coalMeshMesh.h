#pragma once
#include <Eigen/Dense>
#include <memory>
#include <string>

#include "coal/BVH/BVH_model.h"
#include "coal/collision.h"
#include "coal/collision_data.h"
#include "coal/contact_patch.h"
#include "coal/math/transform.h"
#include "coal/mesh_loader/loader.h"

namespace apbd {
// TODO: Probably going to be a compiler issue because this is already declared
// in odeBoxBox.h
struct Contacts {
    // Number of contacts
    int count;
    // Maximum penetration depth
    double depthMax;
    // Penetration depths
    double depths[8];
    // Contact points in world space
    Eigen::Vector3d positions[8];
    // Contact normal (same for all points)
    Eigen::Vector3d normal;
};

std::vector<std::shared_ptr<coal::ConvexBase> > loadConvexDecompositions(const std::string &filename);
std::shared_ptr<coal::ConvexBase> loadConvexMesh(const std::string &filename);

Contacts coalMeshMesh(const Eigen::Matrix4d &M1, const Eigen::Matrix4d &M2,
                      std::shared_ptr<coal::ConvexBase> shape1, std::shared_ptr<coal::ConvexBase> shape2);

}  // namespace apbd