# https://pypi.org/project/cma/
# TODO: Move this and other scripts to src
import subprocess as sp
import cma
import os
import numpy as np
from tqdm import tqdm
from typing import List
import random

NUM_ENVIRONMENTS = 4096

MODEL_ID         = 442
NUM_SUBSTEPS     = 50
SEED             = 441
MAX_STEPS        = 100
MAX_FEVAL        = NUM_ENVIRONMENTS * MAX_STEPS
DO_WRITE         = False
EXEC_CMD         = ""
CONVERGE_TOL     = 1e-12
DIM              = 6

random.seed(SEED)

DEBUG = False
def printd(msg: str):
    if DEBUG:
        print(f"{msg}")

# Random array of size n with values in [a, b] using set
def genRandArr(n, a, b):
    seen = set()
    while len(seen) != 50:
        seen.add(random.randint(a, b))
    return list(seen)

# Assumes we are running from .../cuda-xpbd/
cwd = os.getcwd()
DEBUG_PATH = os.path.join(cwd, "cmaes/data", "debug.txt")
INP_PATH   = os.path.join(cwd, "cmaes/data", "velocities.txt")
OUT_PATH   = os.path.join(cwd, "cmaes/data", "objective.txt")
EXEC_CMD   = f"{cwd}/hop.sh release_cuda {MODEL_ID} {NUM_ENVIRONMENTS} {NUM_SUBSTEPS} {DO_WRITE}"
printd(cwd)
printd(f"#{DEBUG_PATH}\n#{INP_PATH}\n#{OUT_PATH}")

# DEVSLASHNULL = open(os.devnull, 'w')
DEVSLASHNULL = open(DEBUG_PATH, 'w')
def fitness_distance(x) -> List[float]:
    x_flat = np.array(x).ravel()
    np.savetxt(INP_PATH, x_flat, delimiter=' ', newline=' ', fmt="%.10f")

    # with open(DEBUG_PATH, 'w') as f:
    res = sp.run(EXEC_CMD, shell=True, stderr=DEVSLASHNULL, stdout=DEVSLASHNULL)
    if res.returncode != 0:
        printd(f"# `{EXEC_CMD}` failed with exit code {res.returncode}")
        raise RuntimeError(f"Exec cmd `{EXEC_CMD}` failed with exit code {res.returncode}")
    
    objectives = []
    with open(OUT_PATH, 'r') as fin:
        output = fin.readline().split(' ')[:-1]
        objectives = list(map(float, output))
    
    printd(f"# Objectives = {objectives}")

    return objectives

"""
Takes in indices[] and runs a script with those values. Runs script with 1 environment len(indices) times
and stores the final step in data/sample50.txt for each run.
"""
STEP_STARTIDX = -16
STEP_ENDIDX   = -4
TEST_EXEC_CMD = f"{cwd}/hop.sh debug_cuda {MODEL_ID} 1 {NUM_SUBSTEPS} {DO_WRITE} > tx.txt"
def sample_outputs(indices: List[int], velocities: List[float]) -> None:
    TXT_PATH = os.path.join(cwd, "tx.txt")

    output = [""] * len(indices)
    for j, idx in enumerate(indices):
        # write current velocity
        cur_vel = velocities[idx]
        v_flat  = np.array(cur_vel).ravel()
        np.savetxt(INP_PATH, v_flat, delimiter=' ', newline=' ', fmt="%.10f")

        # print(f"Running {TEST_EXEC_CMD}")
        res = sp.run(TEST_EXEC_CMD, shell=True)
        if res.returncode != 0:
            printd(f"# `{TEST_EXEC_CMD}` failed with exit code {res.returncode}")
            raise RuntimeError(f"Exec cmd `{TEST_EXEC_CMD}` failed with exit code {res.returncode}")

        with open(TXT_PATH, 'r') as fin:
            lines = fin.readlines()[STEP_STARTIDX:STEP_ENDIDX]
            output[j] = (''.join(lines))

    # write output back
    with open(TXT_PATH, 'w') as fin:
        fin.write(''.join(output))

    # print(output)
    # exit(1)
# GUESS_VEC = list( (goal - origin) / np.linalg.norm(goal - origin) )
# 10 10 10 25 0 160
# 10.4036928110 10.0484809075 10.0851058919 30.3226707817 0.2016256760 169.9435648538
# 9.6847074055 9.7142719089 9.7547234550 29.9624322641 0.0050233115 170.2891737905
GUESS_VEC = list(np.array([10, 10, 10, 30, 0, 170]))
print(f"# Initial Guess: {GUESS_VEC}")

# keep wx0, wy0, wz0 as none for now
x0 = GUESS_VEC
print(x0)
sigma0 = 0.1
opts = cma.CMAOptions()
opts.set('tolfunhist', -1)
# opts.set('tolfun', -1)
opts.set('tolfacupx', 100000);
opts.set('tolflatfit', MAX_STEPS)
opts.set("maxfevals", MAX_FEVAL)
opts.set("ftarget", CONVERGE_TOL)
opts.set("seed", SEED)
# opts.set("bounds", [-np.inf, np.inf])
opts.set("popsize", NUM_ENVIRONMENTS)
opts.set("verb_log", 0)
opts.set("verb_disp", 0)
opts.set("verb_plot", 0)
opts.set("verbose", 0)
opts.set("verb_log_expensive", 0)
opts.set("verb_filenameprefix", "")

es = cma.CMAEvolutionStrategy(x0, sigma0, options=opts)

print(f"### RUNNING ###")
step = 0
pbar = tqdm(total=None, desc="Descent", unit=f" step")
min_idx, min_val = -1, float('inf')
best_human = None
while not es.stop():
    # We should run 1 fitness function that uses each asked value, and we can return all of them.
    solns = es.ask(number=NUM_ENVIRONMENTS)
    objectives = fitness_distance(solns)
    es.tell(solns, objectives)

    # batch min
    for i, v in enumerate(objectives):
        if v < min_val:
            # (human readable)
            min_val = v
            min_idx = i
            best_human = ' '.join(f"{v:.10f}" for v in solns[min_idx])

    # rand_50s = genRandArr(50, 0, NUM_ENVIRONMENTS - 1)
    # sample_outputs(rand_50s, solns)
    
    print(f" Best batch idx, val = ({min_idx}, {min_val})")
    print(f"Human Readable: {best_human}")

    step += 1
    pbar.update(1)

DEVSLASHNULL.close()

print(f"### RESULTS ###")
es.result_pretty()
soln = ' '.join(f"{v:.10f}" for v in es.result.xbest)
print(f"# Best Output: {soln}")