// https://github.com/CMA-ES/libcmaes?tab=readme-ov-file#sample-code
#include <libcmaes/cmaes.h>

#include <boost/filesystem.hpp>
#include <iostream>

using std::vector, std::string, std::cout, std::endl;
using namespace libcmaes;
namespace fs = boost::filesystem;

// Number of worlds to simulate in parallel, should
const int NUM_ENVIRONMENTS = 1;
const int NUM_SUBSTEPS     = 20;
string EXEC_CMD;

#define ERR_STR string(__FILE__) + ":" + std::to_string(__LINE__)


/*
Note that angular velocity is first, not linear. We can have 6 * NUM_SCENES:

wx0 wy0 wz0 vx0 vy0 vz0 | wx1 wy1 wz1 vx1 vy1 vz1 | ...

From here we can start a subshell that runs the simulation, assuming some
baseline # of environments
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
    // printf("# Executing EXEC_CMD: %s\n", EXEC_CMD.c_str());
    if (std::system(EXEC_CMD.c_str()) != 0) {
        throw std::runtime_error("Exec failed at " + ERR_STR);
    }

    // std::system is synchronous; so we can assume it writes back the dp'dp for
    // us into data/objective by now
    std::ifstream fin("./data/objective.txt");

    float objective;
    fin >> objective;
    fin.close();

    return objective;
};

int main(int argc, char *argv[]) {
    // https://en.cppreference.com/w/cpp/filesystem/current_path
    // (technically using Boost bc C++11 but that's ok)
    fs::path cwd = fs::current_path();

    string DEBUG_PATH = fs::absolute("data/debug.txt").string();
    string INP_PATH = fs::absolute("data/objective.txt").string();
    string OUT_PATH = fs::absolute("data/velocities.txt").string();

    EXEC_CMD = cwd.string() + "/../hop.sh debug_cuda";

    // Add flags
    EXEC_CMD += " 99 " + std::to_string(NUM_ENVIRONMENTS) + " " +
                std::to_string(NUM_SUBSTEPS) + " > " + DEBUG_PATH + " 2>&1";

    // 6 dims, 3 for linear vel, 3 for angular. If we awnted to vary the number
    // of bodies, we would do * N, but we only modify the lowest (1st) bo
    const int DIM = 6;
    std::vector<double> x0(DIM * NUM_ENVIRONMENTS, 10.0f);
    float sigma = 0.1f;

    const uint64_t SEED = 441;
    const float THRESHOLD = 1e-5f;
    CMAParameters<> cmparams(x0, sigma);
    cmparams.set_seed(SEED);
    cmparams.set_mt_feval(false);
    cmparams.set_max_iter(1000);
    cmparams.set_ftolerance(THRESHOLD);

    CMASolutions cmasols = cmaes<>(distance, cmparams);
    cout << "Best Solution: " << cmasols << endl;
    std::cout << "optimization took " << cmasols.elapsed_time() / 1000.0
              << " seconds\n";

    return cmasols.run_status();
}