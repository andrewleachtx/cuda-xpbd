# https://pypi.org/project/cma/
import subprocess as sp
import cma
import os
import numpy as np
from tqdm import tqdm
import time
from typing import List

NUM_ENVIRONMENTS = 2048

MODEL_ID         = 442
NUM_SUBSTEPS     = 100
SEED             = 441
MAX_STEPS        = 100
MAX_FEVAL        = NUM_ENVIRONMENTS * MAX_STEPS
DO_WRITE         = False
EXEC_CMD         = ""
CONVERGE_TOL     = 1e-12
DIM              = 6

DEBUG = False
def printd(msg: str):
    if DEBUG:
        print(f"{msg}")

# Assumes we are running from .../cuda-xpbd/
cwd = os.getcwd()
DEBUG_PATH = os.path.join(cwd, "cmaes/data", "debug.txt")
INP_PATH   = os.path.join(cwd, "cmaes/data", "velocities.txt")
OUT_PATH   = os.path.join(cwd, "cmaes/data", "objective.txt")
EXEC_CMD   = f"{cwd}/hop.sh release_cuda {MODEL_ID} {NUM_ENVIRONMENTS} {NUM_SUBSTEPS} {DO_WRITE}"
printd(cwd)
printd(f"#{DEBUG_PATH}\n#{INP_PATH}\n#{OUT_PATH}")

DEVSLASHNULL = open(os.devnull, 'w')
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

# GUESS_VEC = list( (goal - origin) / np.linalg.norm(goal - origin) )
# -1.0464412582 2.5312968977 -1.8802998887 -20.6529659668 0.0 186.4705882562
GUESS_VEC = list(np.array([20, 0, 200]))
print(f"# Initial Guess: {GUESS_VEC}")

# keep wx0, wy0, wz0 as none for now
x0 = [0.0, 0.0, 0.0] + GUESS_VEC
print(x0)
sigma0 = 0.1
opts = cma.CMAOptions()
opts.set('tolfunhist', -1)
# opts.set('tolfun', -1)
opts.set('tolflatfit', MAX_STEPS)
opts.set("maxfevals", MAX_FEVAL)
opts.set("ftarget", CONVERGE_TOL)
opts.set("seed", SEED)
opts.set("bounds", [-np.inf, np.inf])
opts.set("popsize", NUM_ENVIRONMENTS)
opts.set("verb_log", 0)
opts.set("verb_disp", 0)
opts.set("verbose", 0)
opts.set("verb_log_expensive", 0)
opts.set("verb_filenameprefix", "")

es = cma.CMAEvolutionStrategy(x0, sigma0, options=opts)

print(f"### RUNNING ###")
step = 0
pbar = tqdm(total=None, desc="Descent", unit=f" step")
min_idx, min_val = -1, float('inf')
while not es.stop():
    # We should run 1 fitness function that uses each asked value, and we can return all of them.
    solns = es.ask(number=NUM_ENVIRONMENTS)
    objectives = fitness_distance(solns)
    es.tell(solns, objectives)

    # Batch min
    # min_idx, min_val = -1, float('inf')
    for i, v in enumerate(objectives):
        if v < min_val:
            min_val = v
            min_idx = i
    
    # 0.1562725026 240.9189158924 -71.8697619566 80.4508945356 -0.0479658356 186.5888997418
    print(f" Best batch idx, val = ({min_idx}, {min_val})")
    #TODO: Logically this isn't right, store the velocity
    human_readable = ' '.join(f"{v:.10f}" for v in solns[min_idx])
    print(f"Human Readable: {human_readable}")

    step += 1
    pbar.update(1)

print(f"### RESULTS ###")
es.result_pretty()
soln = ' '.join(f"{v:.10f}" for v in es.result.xbest)
print(f"### Best Output: {soln}")
