#pragma once

#include "apbd/Model.h"
#include "apbd/ShapeMeshObj.h"
#include "se3/lib.h"
#include "util.h"
#include <math.h>

apbd::Model createModelSample(int modelID, float h, unsigned int substeps,
                              apbd::Body *&bodies, size_t scene_count)
{
    auto model = apbd::Model();

    switch (modelID)
    {
        // DEBUGGING TEST CASE
        case -1:
        {
            model.h = 0.005;
            model.tEnd = 0.1;
            model.substeps = 10;
            model.forward_iters = 5;
            model.reverse_iters = 25;
            float density = 1.0;
            float w = 1;
            Eigen::Vector3f sides{w, w, w};
            model.gravity = Eigen::Vector3f(0, 0, -980).transpose();
            model.ground_E = Eigen::Matrix4f::Identity();
            float mu = 0.1;

            model.ground_size = 20;

            model.body_count = 2;
            model.bodies = new apbd::BodyReference[2];
            bodies = new apbd::Body[2];
            bodies[0] = apbd::Body(
                apbd::BodyRigid(apbd::ShapeCuboid{sides}, density, true, mu));
            bodies[1] = apbd::Body(
                apbd::BodyRigid(apbd::ShapeCuboid{sides}, density, true, mu));

            Eigen::Matrix4f E = Eigen::Matrix4f::Identity();
            Eigen::Matrix3f R = se3::aaToMat(Eigen::Vector3f(1, 1, 1), M_PI / 4);
            E.block<3, 3>(0, 0) = R;
            E.block<3, 1>(0, 3) = Eigen::Vector3f(0, 0, 0.75);
            bodies[0].setInitTransform(E);
            E.block<3, 1>(0, 3) = Eigen::Vector3f(0, 0.0, 2);
            bodies[1].setInitTransform(E);
            break;
        }
        case 1:
        {
            // Stacking: 10 rigid bodies with offset
            model.tEnd = 1;
            model.h = h;
            model.substeps = substeps;
            model.forward_iters = 5;
            model.reverse_iters = 25;
            float density = 1.0;
            float w = 1;
            Eigen::Vector3f sides{w, w, w};
            model.gravity = Eigen::Vector3f(0, 0, -980).transpose();
            model.ground_E = Eigen::Matrix4f::Identity();
            float mu = 0.5;

            model.ground_size = 20;

            size_t n = 10;
            bodies = new apbd::Body[n];
            model.body_count = n;
            model.bodies = new apbd::BodyReference[n];
            for (size_t i = 0; i < n; i++)
            {
                bodies[i] = apbd::Body(
                    apbd::BodyRigid(apbd::ShapeCuboid{sides}, density, true, mu));
                Eigen::Matrix4f E = Eigen::Matrix4f::Identity();
                float x = 0.04 * (i + 1);
                float y = 0;
                float z = (i + 0.5) * w;
                E.block<3, 1>(0, 3) = Eigen::Vector3f(x, y, z);
                bodies[i].setInitTransform(E);
                if (i == 1)
                {
                    bodies[i].setInitVelocity(Eigen::Matrix<float, 6, 1>(0, 0, 0, 0, 0, 0));
                }
            }
            break;
        }
        case 2:
        {
            // Stacking: 2 rigid bodies without friction
            model.tEnd = 1;
            model.h = h;
            model.substeps = substeps;
            model.forward_iters = 5;
            model.reverse_iters = 25;
            float density = 1.0;
            float w = 1;
            Eigen::Vector3f sides{w, w, w};
            model.gravity = Eigen::Vector3f(0, 0, -980).transpose();
            model.ground_E = Eigen::Matrix4f::Identity();
            float mu = 0.0;

            model.ground_size = 20;

            size_t n = 2;
            bodies = new apbd::Body[n];
            model.body_count = n;
            model.bodies = new apbd::BodyReference[n];
            for (size_t i = 0; i < n; i++)
            {
                bodies[i] = apbd::Body(
                    apbd::BodyRigid(apbd::ShapeCuboid{sides}, density, true, mu));
                Eigen::Matrix4f E = Eigen::Matrix4f::Identity();
                float x = 0.5 * (i + 1);
                float y = 0;
                float z = (i + 0.5) * w;
                E.block<3, 1>(0, 3) = Eigen::Vector3f(x, y, z);
                bodies[i].setInitTransform(E);
                if (i == 1)
                {
                    bodies[i].setInitVelocity(Eigen::Matrix<float, 6, 1>(0, 0, 0, 0, 0, 0));
                }
            }
            break;
        }
        case 3:
        {
            // Dynamic Fricition: sliding distance test
            model.tEnd = 1;
            model.h = h;
            model.substeps = substeps;
            model.forward_iters = 5;
            model.reverse_iters = 25;
            float density = 1.0;
            float w = 1;
            Eigen::Vector3f sides{w, w, w};
            model.gravity = Eigen::Vector3f(0, 0, -980).transpose();
            model.ground_E = Eigen::Matrix4f::Identity();
            float mu = 0.9;

            model.ground_size = 20;

            size_t n = 1;
            bodies = new apbd::Body[n];
            model.body_count = n;
            model.bodies = new apbd::BodyReference[n];
            for (size_t i = 0; i < n; i++)
            {
                bodies[i] = apbd::Body(
                    apbd::BodyRigid(apbd::ShapeCuboid{sides}, density, true, mu));
                Eigen::Matrix4f E = Eigen::Matrix4f::Identity();
                float x = 0.5 * (i + 1);
                float y = 0;
                float z = (i + 0.5) * w;
                E.block<3, 1>(0, 3) = Eigen::Vector3f(x, y, z);
                bodies[i].setInitTransform(E);
                if (i == 1)
                {
                    bodies[i].setInitVelocity(
                        Eigen::Matrix<float, 6, 1>(0, 0, 0, 100, 0, 0));
                }
            }
            break;
        }
        case 4:
        {
            // Static Friciton: 2 rigid bodies on a slope
            model.tEnd = 1;
            model.h = h;
            model.substeps = substeps;
            model.forward_iters = 5;
            model.reverse_iters = 25;
            float density = 1.0;
            float w = 1;
            Eigen::Vector3f sides{w, w, w};
            model.gravity = Eigen::Vector3f(0, 0, -980).transpose();
            model.ground_E = Eigen::Matrix4f::Identity();

            float angle = 20.0 * M_PI / 180.0;
            Eigen::Matrix3f R = se3::aaToMat(Eigen::Vector3f(0, 1, 0), angle);
            model.ground_E.block<3, 3>(0, 0) = R;
            float mu = 1.01 * (sin(angle) / cos(angle));

            model.ground_size = 20;

            size_t n = 2;
            bodies = new apbd::Body[n];
            model.body_count = n;
            model.bodies = new apbd::BodyReference[n];
            for (size_t i = 0; i < n; i++)
            {
                bodies[i] = apbd::Body(
                    apbd::BodyRigid(apbd::ShapeCuboid{sides}, density, true, mu));
                Eigen::Matrix4f E = Eigen::Matrix4f::Identity();
                E.block<3, 3>(0, 0) = R;
                float x = 0.0;
                float y = 0;
                float z = (i + 0.5) * w;
                E.block<3, 1>(0, 3) = R * Eigen::Vector3f(x, y, z);
                bodies[i].setInitTransform(E);
                if (i == 1)
                {
                    bodies[i].setInitVelocity(Eigen::Matrix<float, 6, 1>(0, 0, 0, 0, 0, 0));
                }
            }
            break;
        }
        case 5:
        {
            // Static Friciton: 10 rigid bodies on a slope
            model.tEnd = 1;
            model.h = h;
            model.substeps = substeps;
            model.forward_iters = 5;
            model.reverse_iters = 25;
            float density = 1.0;
            float w = 1;
            Eigen::Vector3f sides{w, w, w};
            model.gravity = Eigen::Vector3f(0, 0, -980).transpose();
            model.ground_E = Eigen::Matrix4f::Identity();

            float angle = 20.0 * M_PI / 180.0;
            Eigen::Matrix3f R = se3::aaToMat(Eigen::Vector3f(0, 1, 0), angle);
            model.ground_E.block<3, 3>(0, 0) = R;
            float mu = 1.01 * (sin(angle) / cos(angle));

            model.ground_size = 20;

            size_t n = 10;
            bodies = new apbd::Body[n];
            model.body_count = n;
            model.bodies = new apbd::BodyReference[n];
            for (size_t i = 0; i < n; i++)
            {
                bodies[i] = apbd::Body(
                    apbd::BodyRigid(apbd::ShapeCuboid{sides}, density, true, mu));
                Eigen::Matrix4f E = Eigen::Matrix4f::Identity();
                E.block<3, 3>(0, 0) = R;
                float x = 0.0;
                float y = 0;
                float z = (i + 0.5) * w;
                E.block<3, 1>(0, 3) = R * Eigen::Vector3f(x, y, z);
                bodies[i].setInitTransform(E);
                if (i == 1)
                {
                    bodies[i].setInitVelocity(Eigen::Matrix<float, 6, 1>(0, 0, 0, 0, 0, 0));
                }
            }
            break;
        }
        case 6:
        {
            // Stacking: 10 rigid bodies falling one by one
            model.tEnd = 1;
            model.h = h;
            model.substeps = substeps;
            model.forward_iters = 5;
            model.reverse_iters = 25;
            float density = 1.0;
            float w = 1;
            Eigen::Vector3f sides{w, w, w};
            model.gravity = Eigen::Vector3f(0, 0, -980).transpose();
            model.ground_E = Eigen::Matrix4f::Identity();
            float mu = 0.5;

            model.ground_size = 20;

            size_t n = 10;
            bodies = new apbd::Body[n];
            model.body_count = n;
            model.bodies = new apbd::BodyReference[n];
            for (size_t i = 0; i < n; i++)
            {
                bodies[i] = apbd::Body(
                    apbd::BodyRigid(apbd::ShapeCuboid{sides}, density, true, mu));
                Eigen::Matrix4f E = Eigen::Matrix4f::Identity();
                float x = 0.04 * (i + 1);
                float y = 0;
                float z = (i + 0.5 + i * 0.1) * w;
                E.block<3, 1>(0, 3) = Eigen::Vector3f(x, y, z);
                bodies[i].setInitTransform(E);
                if (i == 1)
                {
                    bodies[i].setInitVelocity(Eigen::Matrix<float, 6, 1>(0, 0, 0, 0, 0, 0));
                }
            }
            break;
        }
        case 7:
        {
            // Stacking: 10 rigidbodies and push the second one
            model.tEnd = 1;
            model.h = h;
            model.substeps = substeps;
            model.forward_iters = 5;
            model.reverse_iters = 25;
            float density = 1.0;
            float w = 1;
            Eigen::Vector3f sides{w, w, w};
            model.gravity = Eigen::Vector3f(0, 0, -980).transpose();
            model.ground_E = Eigen::Matrix4f::Identity();
            float mu = 0.5;

            model.ground_size = 20;

            size_t n = 10;
            bodies = new apbd::Body[n];
            model.body_count = n;
            model.bodies = new apbd::BodyReference[n];
            for (size_t i = 0; i < n; i++)
            {
                bodies[i] = apbd::Body(
                    apbd::BodyRigid(apbd::ShapeCuboid{sides}, density, true, mu));
                Eigen::Matrix4f E = Eigen::Matrix4f::Identity();
                float x = 0.04 * (i + 1);
                float y = 0;
                float z = (i + 0.5) * w;
                E.block<3, 1>(0, 3) = Eigen::Vector3f(x, y, z);
                bodies[i].setInitTransform(E);
                if (i == 1)
                {
                    bodies[i].setInitVelocity(
                        Eigen::Matrix<float, 6, 1>(0, 0, 0, 100, 0, 0));
                }
            }
            break;
        }
        case 8:
        {
            // Stacking: 10 rigidbodies with large offset
            model.tEnd = 1;
            model.h = h;
            model.substeps = substeps;
            model.forward_iters = 5;
            model.reverse_iters = 25;
            float density = 1.0;
            float w = 1;
            Eigen::Vector3f sides{w, w, w};
            model.gravity = Eigen::Vector3f(0, 0, -980).transpose();
            model.ground_E = Eigen::Matrix4f::Identity();
            float mu = 0.5;

            model.ground_size = 20;

            size_t n = 10;
            bodies = new apbd::Body[n];
            model.body_count = n;
            model.bodies = new apbd::BodyReference[n];
            for (size_t i = 0; i < n; i++)
            {
                bodies[i] = apbd::Body(
                    apbd::BodyRigid(apbd::ShapeCuboid{sides}, density, true, mu));
                Eigen::Matrix4f E = Eigen::Matrix4f::Identity();
                float x = 0.4 * (i + 1);
                float y = 0;
                float z = (i + 0.5) * w;
                E.block<3, 1>(0, 3) = Eigen::Vector3f(x, y, z);
                bodies[i].setInitTransform(E);
                if (i == 1)
                {
                    bodies[i].setInitVelocity(Eigen::Matrix<float, 6, 1>(0, 0, 0, 0, 0, 0));
                }
            }
            break;
        }
        case 9:
        {
            // Stacking: 10 rigidbodies tetris
            model.tEnd = 1;
            model.h = h;
            model.substeps = substeps;
            model.forward_iters = 5;
            model.reverse_iters = 25;
            float density = 1.0;
            float w = 1;
            Eigen::Vector3f sides{w, w, w};
            model.gravity = Eigen::Vector3f(0, 0, -980).transpose();
            model.ground_E = Eigen::Matrix4f::Identity();
            float mu = 0.5;

            model.ground_size = 20;

            size_t n = 10;
            bodies = new apbd::Body[n];
            model.body_count = n;
            model.bodies = new apbd::BodyReference[n];
            for (size_t i = 0; i < n; i++)
            {
                bodies[i] = apbd::Body(
                    apbd::BodyRigid(apbd::ShapeCuboid{sides}, density, true, mu));
                Eigen::Matrix4f E = Eigen::Matrix4f::Identity();
                float x = 0.04 * (i + 1);
                float y = 0;
                float z = (i + 0.5) * w + 1;
                E.block<3, 1>(0, 3) = Eigen::Vector3f(x, y, z);
                bodies[i].setInitTransform(E);
                if (i == 1)
                {
                    bodies[i].setInitVelocity(Eigen::Matrix<float, 6, 1>(0, 0, 0, 0, 0, 0));
                }
            }
            break;
        }
        case 10:
        {
            // Stacking: the wall
            model.tEnd = 1;
            model.h = h;
            model.substeps = substeps;
            model.forward_iters = 5;
            model.reverse_iters = 25;
            float density = 1.0;
            float w = 1;
            Eigen::Vector3f sides{w, w, w};
            model.gravity = Eigen::Vector3f(0, 0, -980).transpose();
            model.ground_E = Eigen::Matrix4f::Identity();
            float mu = 0.5;

            model.ground_size = 20;

            size_t n = 10;
            bodies = new apbd::Body[n];
            model.body_count = n;
            model.bodies = new apbd::BodyReference[n];
            for (size_t i = 0; i < 5; i++)
            {
                for (size_t j = 0; j < 2; j++)
                {
                    bodies[i * 2 + j] = apbd::Body(
                        apbd::BodyRigid(apbd::ShapeCuboid{sides}, density, true, mu));
                    Eigen::Matrix4f E = Eigen::Matrix4f::Identity();
                    float x = ((i + 2) % 2) * 0.3 * w + (j + 0.5) * w - w;
                    float y = 0;
                    float z = (i + 0.5) * w;
                    E.block<3, 1>(0, 3) = Eigen::Vector3f(x, y, z);
                    bodies[i * 2 + j].setInitTransform(E);
                }
            }
            break;
        }
        case 11:
        {
            // Stacking: heavy head
            model.tEnd = 1;
            model.h = h;
            model.substeps = substeps;
            model.forward_iters = 5;
            model.reverse_iters = 25;
            float w = 1;
            Eigen::Vector3f sides{w, w, w};
            model.gravity = Eigen::Vector3f(0, 0, -980).transpose();
            model.ground_E = Eigen::Matrix4f::Identity();
            float mu = 0.5;

            model.ground_size = 20;

            size_t n = 10;
            bodies = new apbd::Body[n];
            model.body_count = n;
            model.bodies = new apbd::BodyReference[n];
            for (size_t i = 0; i < n; i++)
            {
                float density = powf(2.0, float(i));
                bodies[i] = apbd::Body(
                    apbd::BodyRigid(apbd::ShapeCuboid{sides}, density, true, mu));
                Eigen::Matrix4f E = Eigen::Matrix4f::Identity();
                float x = 0.04 * (i + 1);
                float y = 0;
                float z = (i + 0.5) * w;
                E.block<3, 1>(0, 3) = Eigen::Vector3f(x, y, z);
                bodies[i].setInitTransform(E);
                if (i == 1)
                {
                    bodies[i].setInitVelocity(Eigen::Matrix<float, 6, 1>(0, 0, 0, 0, 0, 0));
                }
            }
            break;
        }
        case 12:
        {
            // Stacking: heavy head 2
            model.tEnd = 1;
            model.h = h;
            model.substeps = substeps;
            model.forward_iters = 5;
            model.reverse_iters = 25;
            float density = 1.0;
            float w = 1;
            Eigen::Vector3f sides{w, w, w};
            model.gravity = Eigen::Vector3f(0, 0, -980).transpose();
            model.ground_E = Eigen::Matrix4f::Identity();
            float mu = 0.5;

            model.ground_size = 20;

            size_t n = 10;
            bodies = new apbd::Body[n];
            model.body_count = n;
            model.bodies = new apbd::BodyReference[n];
            for (size_t i = 0; i < n; i++)
            {
                float scale = powf(2.0, float(i));
                bodies[i] = apbd::Body(
                    apbd::BodyRigid(apbd::ShapeCuboid{scale * sides}, density, true, mu));
                Eigen::Matrix4f E = Eigen::Matrix4f::Identity();
                float x = 0.0;
                float y = 0;
                float z = 0.5 * w * scale + scale - 1;
                E.block<3, 1>(0, 3) = Eigen::Vector3f(x, y, z);
                bodies[i].setInitTransform(E);
                if (i == 1)
                {
                    bodies[i].setInitVelocity(Eigen::Matrix<float, 6, 1>(0, 0, 0, 0, 0, 0));
                }
            }
            break;
        }
        case 13:
        {
            // Stacking: seasaw
            model.tEnd = 1;
            model.h = h;
            model.substeps = substeps;
            model.forward_iters = 5;
            model.reverse_iters = 25;
            float density = 1.0;
            float w = 1;
            Eigen::Vector3f sides{w, w, w};
            model.gravity = Eigen::Vector3f(0, 0, -980).transpose();
            model.ground_E = Eigen::Matrix4f::Identity();
            float mu = 0.5;

            model.ground_size = 10;

            size_t n = 10;
            bodies = new apbd::Body[n];
            model.body_count = n;
            model.bodies = new apbd::BodyReference[n];
            for (size_t j = 0; j < 2; j++)
            {
                for (size_t i = 0; i < 6; i++)
                {
                    bodies[j * 6 + i] = apbd::Body(
                        apbd::BodyRigid(apbd::ShapeCuboid{sides}, density, true, mu));
                    Eigen::Matrix4f E = Eigen::Matrix4f::Identity();
                    float x = 0.0;
                    if (j == 0)
                    {
                        x = 1.5;
                    }
                    else
                    {
                        x = -1.5;
                    }
                    float y = 0;
                    float z = (i + 0.5) * w + 2;
                    E.block<3, 1>(0, 3) = Eigen::Vector3f(x, y, z);
                    bodies[j * 6 + i].setInitTransform(E);
                    if (i == 1 && j == 1)
                    {
                        break;
                    }
                }
            }
            bodies[8] = apbd::Body(
                apbd::BodyRigid(apbd::ShapeCuboid{sides}, density, true, mu));
            Eigen::Matrix4f E = Eigen::Matrix4f::Identity();
            E(2, 3) = 0.5;
            bodies[8].setInitTransform(E);
            bodies[9] = apbd::Body(apbd::BodyRigid(
                apbd::ShapeCuboid{Eigen::Vector3f(4, 1, 1)}, density, true, mu));
            E = Eigen::Matrix4f::Identity();
            E(2, 3) = 1.5;
            bodies[9].setInitTransform(E);
            break;
        }
        case 14:
        {
            // Stacking: seasaw 2
            model.tEnd = 1;
            model.h = h;
            model.substeps = substeps;
            model.forward_iters = 5;
            model.reverse_iters = 25;
            float density = 1.0;
            float w = 1;
            Eigen::Vector3f sides{w, w, w};
            model.gravity = Eigen::Vector3f(0, 0, -980).transpose();
            model.ground_E = Eigen::Matrix4f::Identity();
            float mu = 0.5;

            model.ground_size = 10;

            size_t n = 14;
            bodies = new apbd::Body[n];
            model.body_count = n;
            model.bodies = new apbd::BodyReference[n];
            for (size_t j = 0; j < 2; j++)
            {
                for (size_t i = 0; i < 6; i++)
                {
                    bodies[j * 6 + i] = apbd::Body(
                        apbd::BodyRigid(apbd::ShapeCuboid{sides}, density, true, mu));
                    Eigen::Matrix4f E = Eigen::Matrix4f::Identity();
                    float x = 0.0;
                    if (j == 0)
                    {
                        x = 1.5;
                    }
                    else
                    {
                        x = -1.5;
                    }
                    float y = 0;
                    float z = (i + 0.5) * w + 2;
                    E.block<3, 1>(0, 3) = Eigen::Vector3f(x, y, z);
                    bodies[j * 6 + i].setInitTransform(E);
                }
            }
            bodies[12] = apbd::Body(
                apbd::BodyRigid(apbd::ShapeCuboid{sides}, density, true, mu));
            Eigen::Matrix4f E = Eigen::Matrix4f::Identity();
            E(2, 3) = 0.5;
            bodies[12].setInitTransform(E);
            bodies[13] = apbd::Body(apbd::BodyRigid(
                apbd::ShapeCuboid{Eigen::Vector3f(4, 1, 1)}, density, true, mu));
            E = Eigen::Matrix4f::Identity();
            E(2, 3) = 1.5;
            bodies[13].setInitTransform(E);
            break;
        }
        /*
            CASES 15-17 CAN BE ADDED, BUT FOR NOW I'M JUST DOING 18 AS THAT IS THE ONE REFERENCED
        */
        case 18:
        {
            // Stacking: Mesh
            // substeps = 1;
            model.tEnd = 1.0f;
            model.h = h;
            model.substeps = substeps;
            model.forward_iters = substeps;
            model.reverse_iters = 25;
            float density = 1.0f;
            float w = 1.0f;
            model.gravity = Eigen::Vector3f(0, 0, -980).transpose();
            model.ground_E = Eigen::Matrix4f::Identity();
            float mu = 0.5f;

            model.ground_size = 20;
            
            float angle = -90.0f * static_cast<float>(M_PI) / 180.0f;
            apbd::ShapeMeshObj mesh = apbd::ShapeMeshObj("./resources/bunny.obj");
            // This function on first intuition is pointless, however it actually populates many member variables
            mesh.computeInertia(density);

            // There are four total bodies (bunnies) for the scene
            size_t n = 2;
            size_t total_bodies = n + 2;

            bodies = new apbd::Body[total_bodies];
            model.body_count = total_bodies;
            model.bodies = new apbd::BodyReference[total_bodies];

            for (size_t i = 0; i < n; i++) {
                apbd::BodyRigid br(mesh, density, true, mu);
                bodies[i] = apbd::Body(br);

                auto R = se3::aaToMat(Eigen::Vector3f(0, 0, 1), angle);

                float x = 0.0f;  // 0*w*(i) but always 0 anyway
                float y = 0.75f * w * (static_cast<float>(i) - 1.0f);
                float z = -0.2f * w;

                Eigen::Matrix4f E = Eigen::Matrix4f::Identity();
                E.block<3,3>(0,0) = R;
                Eigen::Vector3f pos = {x, y, z};
                E.block<3,1>(0,3) = R * pos;

                bodies[i].setInitTransform(E * mesh.E_oi);
            }

            // Third bunny
            bodies[2] = apbd::Body(apbd::BodyRigid(mesh, density, true, mu));
            auto R = se3::aaToMat(Eigen::Vector3f(0, 0, 1), angle);
            Eigen::Matrix4f E = Eigen::Matrix4f::Identity();
            float x = -1.25f;
            float y = -0.75f;
            float z = -0.2f * w;
            Eigen::Vector3f pos = {x, y, z};
            E.block<3, 3>(0, 0) = R;
            E.block<3, 1>(0, 3) = R * pos;
            bodies[2].setInitTransform(E * mesh.E_oi);

            // Fourth bunny
            bodies[3] = apbd::Body(apbd::BodyRigid(mesh, density, true, mu));
            R = se3::aaToMat(Eigen::Vector3f(0, 0, 1), 0.0f);
            E = Eigen::Matrix4f::Identity();
            x = -1.0f;
            y = 0.0f;
            z = 1.0f * w;
            pos = {x, y, z};
            E.block<3, 3>(0, 0) = R;
            E.block<3, 1>(0, 3) = R * pos;
            bodies[3].setInitTransform(E);
        }
    }

    model.init();
    model.create_store(scene_count);

    return model;
}
