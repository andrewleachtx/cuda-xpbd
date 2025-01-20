# https://pypi.org/project/cma/
import subprocess as sp
import cma
import os
import numpy as np
from tqdm import tqdm
import time
from typing import List

NUM_ENVIRONMENTS = 2048
DIM              = 6
GOAL_X           = 8.0
GOAL_Y           = 0.0
GOAL_Z           = 0.5

NUM_SUBSTEPS     = 10
MODEL_ID         = 100
SEED             = 441
CONVERGE_TOL     = 1e-12
MAX_ITER         = 100
EXEC_CMD         = ""

DEBUG = False
def printd(msg: str):
    if DEBUG:
        print(f"{msg}")

# Assumes we are running from .../cuda-xpbd/
cwd = os.getcwd()
DEBUG_PATH = os.path.join(cwd, "cmaes/data", "debug.txt")
INP_PATH   = os.path.join(cwd, "cmaes/data", "velocities.txt")
OUT_PATH   = os.path.join(cwd, "cmaes/data", "objective.txt")
EXEC_CMD   = f"{cwd}/hop.sh release_cuda {MODEL_ID} {NUM_ENVIRONMENTS} {NUM_SUBSTEPS}"
printd(cwd)
printd(f"#{DEBUG_PATH}\n#{INP_PATH}\n#{OUT_PATH}")

def fitness_distance(x) -> List[float]:
    # TODO: May need to truncate but probably not
    # Flatten logic could be faster
    with open(INP_PATH, 'w') as fout:
        for vel in x:
            fout.write(' '.join(map(str, vel)))

    with open(DEBUG_PATH, 'w') as f:
        res = sp.run(EXEC_CMD, shell=True, stderr=sp.STDOUT, stdout=f)
        if res.returncode != 0:
            printd(f"# `{EXEC_CMD}` failed with exit code {res.returncode}")
            raise RuntimeError(f"Exec cmd `{EXEC_CMD}` failed with exit code {res.returncode}")
    
    objectives = []
    with open(OUT_PATH, 'r') as fin:
        objectives = list(map(float, fin.readline().split(' ')))
    
    printd(f"# Objectives = {objectives}")

    return objectives

# Make initial guess based on a unit vector towards the goal, assume cube at 0, 0, 0.5
goal = np.array([GOAL_X, GOAL_Y, GOAL_Z])
origin = np.array([0.0, 0.0, 0.5])

GUESS_VEC = list( (goal - origin) / np.linalg.norm(goal - origin) )
print(f"# Initial Guess: {GUESS_VEC}")

x0 = [0.0, 0.0, 0.0] + GUESS_VEC
sigma0 = 0.1
opts = cma.CMAOptions()
# opts.set('tolfunhist', 1e-12)
# opts.set('tolfun', -1)
opts.set('ftarget', CONVERGE_TOL)
opts.set('seed', SEED)
opts.set('bounds', [-np.inf, np.inf])
opts.set('popsize', NUM_ENVIRONMENTS)

es = cma.CMAEvolutionStrategy(x0, sigma0, options=opts)

# N = 4 + floor(3 * log(DIM))
# N = es.popsize
# solns = es.ask(number=NUM_ENVIRONMENTS)

print(f"### RUNNING ###")
for step in tqdm(range(MAX_ITER)):
    if es.stop():
        break

    printd(f"# Step {step}/{MAX_ITER}")
    solns = es.ask(number=NUM_ENVIRONMENTS)

    # We should run 1 fitness function that uses each asked value, and we can return all of them.
    objectives = fitness_distance(solns)

    es.tell(solns, objectives)

print(f"### RESULTS ###")
es.result_pretty()
