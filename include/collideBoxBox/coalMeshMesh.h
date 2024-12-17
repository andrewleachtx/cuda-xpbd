#pragma once
#include <Eigen/Dense>
#include <string>
#include <memory>
#include "coal/math/transform.h"
#include "coal/mesh_loader/loader.h"
#include "coal/BVH/BVH_model.h"
#include "coal/collision.h"
#include "coal/collision_data.h"
#include "coal/contact_patch.h"

namespace apbd {
    // TODO: Probably going to be a compiler issue because this is already declared in odeBoxBox.h 
    struct Contacts {
        // Number of contacts
        int count;
        // Maximum penetration depth
        double depthMax;
        // Penetration depths
        double depths[8];
        // Contact points in world space
        Eigen::Vector3f positions[8];
        // Contact normal (same for all points)
        Eigen::Vector3f normal;
    };
    
    std::shared_ptr<coal::ConvexBase> loadConvexMesh(const std::string& file_name);

    Contacts coalMeshMesh(const Eigen::Matrix4d& M1,
                          const std::string &meshPath1,
                          const Eigen::Matrix4d& M2,
                          const std::string &meshPath2);

} // namespace apbd