#pragma warning(disable : 4996)
#include <iostream>
#include <memory>

#include "util.h"

#define ENABLE_VHACD_IMPLEMENTATION 1
// #include "VHACD.h"
#include "collideBoxBox/coalMeshMesh.h"

// #define DEBUG_MAIN
//  #define MATLAB_MEX_BUILD

/** If instricuted, compile a mex function for Matlab.  */
// #ifdef MATLAB_MEX_BUILD
// #include "mex.h"
// #define A(i, j) A[i + j * M]
// #else
// #define mexPrintf printf
// #endif

// Function to load a convex mesh from a `.obj`, `.stl` or `.dae` file.
//
// This function imports the object inside the file as a BVHModel, i.e. a point
// cloud which is hierarchically transformed into a tree of bounding volumes.
// The leaves of this tree are the individual points of the point cloud
// stored in the `.obj` file.
// This BVH can then be used for collision detection.
//
// For better computational efficiency, we sometimes prefer to work with
// the convex hull of the point cloud. This insures that the underlying object
// is convex and thus very fast collision detection algorithms such as
// GJK or EPA can be called with this object.
// Consequently, after creating the BVH structure from the point cloud, this
// function also computes its convex hull.

namespace apbd {

std::shared_ptr<coal::ConvexBase> loadConvexMesh(const std::string& filename) {
    coal::NODE_TYPE bv_type = coal::BV_AABB;
    coal::MeshLoader loader(bv_type);
    coal::BVHModelPtr_t bvh = loader.load(filename);
    bvh->buildConvexHull(true, "Qt");
    return bvh->convex;
}

/*
    Because coal does not currently support CVHD
   (https://github.com/coal-library/coal/issues/448) we can do it ourselves.


    Generates a vector of Coal convex bases which represent our convex hull
   decomposition, and stores them in a vector.
*/
// std::vector<std::shared_ptr<coal::ConvexBase>> loadConvexDecompositions(
    // const std::string& filename) {
    // coal::NODE_TYPE bv_type = coal::BV_AABB;
    // coal::MeshLoader loader(bv_type);
    // coal::BVHModelPtr_t bvh_original = loader.load(filename);

    // // Reserve to use for each more efficiently
    // std::vector<float> v_flat;
    // std::vector<uint32_t> t_flat;
    // v_flat.reserve(bvh_original->num_vertices);
    // t_flat.reserve(bvh_original->num_tris);

    // for (const auto& v : *bvh_original->vertices) {
    //     v_flat.push_back(v.x());
    //     v_flat.push_back(v.y());
    //     v_flat.push_back(v.z());
    // }

//     // coal::Triangle v0, v1, v2 accessible with []
//     for (const auto& tri : *bvh_original->tri_indices) {
//         t_flat.push_back(tri[0]);
//         t_flat.push_back(tri[1]);
//         t_flat.push_back(tri[2]);
//     }

//     /*
//         At this point the initial convex hull has been built, and we can use the
//        VHACD interface to work with it

//     https://kmamou.blogspot.com/2014/12/v-hacd-20-in-your-project.html
//     */

//     VHACD::IVHACD::Parameters params;
//     VHACD::IVHACD* interfaceVHACD = VHACD::CreateVHACD();
//     bool res =
//         interfaceVHACD->Compute(v_flat.data(), v_flat.size() / 3, t_flat.data(),
//                                 t_flat.size() / 3, params);

//     if (!res) {
//         TRACE("Failed to compute convex decomposition")
//         exit(1);
//     }

//     uint32_t hull_ct = interfaceVHACD->GetNConvexHulls();
//     std::vector<std::shared_ptr<coal::ConvexBase>> cv_hulls(hull_ct, nullptr);
//     for (uint32_t i = 0; i < hull_ct; i++) {
//         VHACD::IVHACD::ConvexHull cv_hull;
//         interfaceVHACD->GetConvexHull(i, cv_hull);

//         const auto& verts = cv_hull.m_points;
//         const auto& tris = cv_hull.m_triangles;

//         // Regenerate a convex base but for this hull
//         std::shared_ptr<coal::BVHModel<coal::AABB>> cv_base =
//             std::make_shared<coal::BVHModel<coal::AABB>>();
//         cv_base->beginModel(verts.size(), tris.size());

//         // For each triangle, we can access mI0, mI1, mI2 in the verts array
//         for (const auto& tri : tris) {
//             VHACD::Vertex v0(verts[tri.mI0]), v1(verts[tri.mI1]),
//                 v2(verts[tri.mI2]);
//             coal::Vec3s ev0, ev1, ev2;

//             // May be a way to use float but CoalScalar is just a double
//             for (int dim = 0; dim < 3; dim++) {
//                 ev0[dim] = static_cast<coal::CoalScalar>(v0[dim]);
//                 ev1[dim] = static_cast<coal::CoalScalar>(v1[dim]);
//                 ev2[dim] = static_cast<coal::CoalScalar>(v2[dim]);
//             }

//             cv_base->addTriangle(ev0, ev1, ev2);
//         }

//         cv_base->endModel();
//         cv_base->buildConvexHull(true, "Qt");

//         cv_hulls[i] = cv_base->convex;
//     }

//     // Clean up
//     interfaceVHACD->Clean();
//     interfaceVHACD->Release();

//     return cv_hulls;
// }

#ifdef DEBUG_MAIN
int main() {
    // Create the coal shapes.
    // Coal supports many primitive shapes: boxes, spheres, capsules, cylinders,
    // ellipsoids, cones, planes, halfspace and convex meshes (i.e. convex hulls
    // of clouds of points). It also supports BVHs (bounding volumes
    // hierarchies), height-fields and octrees.
    std::shared_ptr<coal::ConvexBase> shape1 = loadConvexMesh("./cube1.obj");
    std::shared_ptr<coal::ConvexBase> shape2 = loadConvexMesh("./cube1.obj");

    // Define the shapes' placement in 3D space
    coal::Transform3s T1;
    T1.setQuatRotation(coal::Quaternion3f::Identity());
    T1.setTranslation(coal::Vec3s(0.0, 0.0, 0.0));
    coal::Transform3s T2 = coal::Transform3s::Identity();
    // T2.setQuatRotation(coal::Quaternion3f::UnitRandom());
    T2.setTranslation(coal::Vec3s(0.2, 0.0, 2.1));

    // Define collision requests and results.
    //
    // The collision request allows to set parameters for the collision pair.
    // For example, we can set a positive or negative security margin.
    // If the distance between the shapes is less than the security margin, the
    // shapes will be considered in collision. Setting a positive security
    // margin can be usefull in motion planning, i.e to prevent shapes from
    // getting too close to one another. In physics simulation, allowing a
    // negative security margin may be usefull to stabilize the simulation.
    coal::CollisionRequest col_req;
    col_req.security_margin = 1e-1;
    // A collision result stores the result of the collision test (signed
    // distance between the shapes, witness points location, normal etc.)
    coal::CollisionResult col_res;

    // Collision call
    coal::collide(shape1.get(), T1, shape2.get(), T2, col_req, col_res);

    coal::ContactPatchRequest patch_req;
    coal::ContactPatchResult patch_res;
    coal::computeContactPatch(shape1.get(), T1, shape2.get(), T2, col_res,
                              patch_req, patch_res);

    // We can access the collision result once it has been populated
    std::cout << "Collision? " << col_res.isCollision() << "\n";
    if (col_res.isCollision()) {
        coal::Contact contact = col_res.getContact(0);
        // The penetration depth does **not** take into account the security
        // margin. Consequently, the penetration depth is the true signed
        // distance which separates the shapes. To have the distance which takes
        // into account the security margin, we can simply add the two together.
        std::cout << "Penetration depth: " << contact.penetration_depth << "\n";
        std::cout
            << "Distance between the shapes including the security margin: "
            << contact.penetration_depth + col_req.security_margin << "\n";
        std::cout << "Witness point on shape1: "
                  << contact.nearest_points[0].transpose() << "\n";
        std::cout << "Witness point on shape2: "
                  << contact.nearest_points[1].transpose() << "\n";
        std::cout << "Normal: " << contact.normal.transpose() << "\n";
    }

    // We can access the collision result once it has been populated
    std::cout << "Contact patch number: " << patch_res.numContactPatches()
              << "\n";
    if (patch_res.numContactPatches() > 0 && col_res.isCollision()) {
        coal::ContactPatch contactpatch = patch_res.getContactPatch(0);

        std::cout << "Penetration depth: " << contactpatch.penetration_depth
                  << "\n";
        std::cout
            << "Distance between the shapes including the security margin: "
            << contactpatch.penetration_depth + col_req.security_margin << "\n";
        for (size_t i = 0; i < contactpatch.size(); ++i) {
            std::cout << "Witness point on shape1: "
                      << (contactpatch.getPoint(i) +
                          0.5 * contactpatch.penetration_depth *
                              contactpatch.getNormal())
                             .transpose()
                      << "\n";
            std::cout << "Witness point on shape2: "
                      << (contactpatch.getPoint(i) -
                          0.5 * contactpatch.penetration_depth *
                              contactpatch.getNormal())
                             .transpose()
                      << "\n";
        }

        std::cout << "Normal: " << contactpatch.getNormal().transpose() << "\n";
    }

    // Before calling another collision test, it is important to clear the
    // previous results stored in the collision result.
    col_res.clear();

    return 0;
}
#endif  // DEBUG

// FIXME: Should shared ptrs be pass by reference?
Contacts coalMeshMesh(const Eigen::Matrix4d& M1, const Eigen::Matrix4d& M2, 
                      std::shared_ptr<coal::ConvexBase> shape1, std::shared_ptr<coal::ConvexBase> shape2) {
    
    coal::Transform3s T1;
    T1.setRotation(M1.topLeftCorner(3, 3));
    T1.setTranslation(M1.topRightCorner(3, 1));
    coal::Transform3s T2;
    T2.setRotation(M2.topLeftCorner(3, 3));
    T2.setTranslation(M2.topRightCorner(3, 1));

    coal::CollisionRequest col_req;
    col_req.security_margin = 1e-1;
    coal::CollisionResult col_res;

    // Collision call
    coal::collide(shape1.get(), T1, shape2.get(), T2, col_req, col_res);

    coal::ContactPatchRequest patch_req;
    coal::ContactPatchResult patch_res;
    patch_req.setPatchTolerance(5e-2);
    patch_req.setNumSamplesCurvedShapes(8);
    coal::computeContactPatch(shape1.get(), T1, shape2.get(), T2, col_res,
                              patch_req, patch_res);

    Contacts results;
    if (patch_res.numContactPatches() > 0 && col_res.isCollision()) {
        coal::ContactPatch contactpatch = patch_res.getContactPatch(0);

        // size_t
        printf("# Detected contactpatch.size() = %lu\n", contactpatch.size());

        results.depthMax = contactpatch.penetration_depth;
        results.count = contactpatch.size();
        if (results.count > 8) results.count = 8;

        for (size_t i = 0; i < contactpatch.size() && i < 8; ++i) {
            results.positions[i] = contactpatch.getPoint(i);
            results.depths[i] = contactpatch.penetration_depth;
        }
        results.normal << contactpatch.getNormal();
    }

    return results;
}

}  // namespace apbd
   // #endif