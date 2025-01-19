// #include "cmaes.h"
#include <libcmaes/cmaes.h>

#include <boost/filesystem.hpp>
#include <iostream>

using std::vector, std::string, std::cout, std::endl;
using namespace libcmaes;
namespace fs = boost::filesystem;

// Number of worlds to simulate in parallel, should
const int NUM_ENVIRONMENTS = 1;
const int NUM_SUBSTEPS = 20;
string EXEC_CMD;

// https://github.com/CMA-ES/libcmaes?tab=readme-ov-file#sample-code

/*
   Let's say that we have a stack of boxes, and our goal is to make this stack
   move to a different location, potentially jumping over an obstacle, by
   adjusting the initial translational and angular velocities of the bottom box.
   In this case, the search space for optimization, x, is the initial velocity
   on the bottom box.

   We can define an objective function f(x) that measures how
   far we are from the goal after simulating for, say, 1 second. For this scene,
   the objective function f(x) should be the squared norm of the distance
   between the center of each box and their goal position: f(x) = dp'*dp, where
   dp = [dp0,dp1,dp2,...] is a vector of position differences for each box. For
   example, dp2 = p2_target - p2, where p2 is the 3D position of the 2nd box,
   and p2_target is the 3D goal position of the 2nd box. This means that dp is
   of size 3*n, where n is the number of bodies. The final objective value,
   dp'*dp is a scalar. Notice that dp depends on x because when the initial
   velocity is changed, the simulation will return a different position vector
   p.

    I should develop or use a stacked scene and vectorize dp, the difference in
   goal and actual position after 1 second given some initial velocity x that
   only applies to the first (lowest) box, and from there I can plug in the f(x)
   = dp'*dp dot product or L2 norm of dp and use CMA-ES to abstract/evaluate
   minimization for an optimal x that minimizes that distance.
*/

/*
The distance function is complicated. We should pass in a vector of 6 * N, where
N is the number of bodies in the stack.

vx0 vy0 vz0 wx0 wy0 wz0 | vx1 vy1 vz1 wx1 wy1 wz1 | ...

From here we can start a subshell that runs the simulation, assuming some
baseline # of environments

../build/hop.sh will run the simulation with the given initial velocities stored
in x, and then we can return the distance from the goal position after 1 second.
*/
FitFunc distance = [](const double *x, const int N) {
    std::ofstream fout("./data/velocities.txt");
    if (!fout.is_open()) {
        throw std::runtime_error("Could not open file for writing velocities");
    }

    for (int i = 0; i < N - 1; i++) {
        fout << x[i] << " ";
    }
    fout << x[N - 1];
    fout.close();

    // Need to fork ../hop.sh
    if (std::system(EXEC_CMD.c_str()) != 0) {
        throw std::runtime_error("Exec failed at " + EXEC_CMD);
    }

    // std::system is synchronous; so we can assume it writes back the dp'dp for
    // us into data/objective by now
    std::ifstream fin("./data/objective.txt");

    float objective;
    fin >> objective;

    return objective;
};

int main(int argc, char *argv[]) {
    // https://en.cppreference.com/w/cpp/filesystem/current_path
    // (technically using Boost bc C++11 but that's ok)
    fs::path cwd = fs::current_path();

    string INP_PATH = fs::absolute("data/objective.txt").string();
    string OUT_PATH = fs::absolute("data/velocities.txt").string();

    string EXEC_CMD = cwd.string() + "/../hop.sh release_cuda";

    // Add flags
    EXEC_CMD += " 99 " + std::to_string(NUM_ENVIRONMENTS) + " " +
                std::to_string(NUM_SUBSTEPS) + " > " + INP_PATH + " 2>&1";

    printf("# Executing EXEC_CMD: %s\n", EXEC_CMD.c_str());
    if (std::system(EXEC_CMD.c_str()) != 0) {
        throw std::runtime_error("Exec failed at " + EXEC_CMD);
    }

    // 6 dims, 3 for linear vel, 3 for angular. If we awnted to vary the number
    // of bodies, we would do * N, but we only modify the lowest (1st) box const
    // int DIM = 6; std::vector<float> x0(DIM * NUM_ENVIRONMENTS, 0.0f); float
    // sigma = 0.1f;

    // CMAParameters<> cmparams(x0, sigma);

    // CMASolutions cmasols = cmaes<>(distance, cmparams);
    // cout << "Best Solution: " << cmasols << endl;
    // std::cout << "optimization took " << cmasols.elapsed_time() / 1000.0
    //           << " seconds\n";

    // return cmasols.run_status();
}