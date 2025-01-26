# https://pypi.org/project/cma/
import subprocess as sp
import cma
import os
import numpy as np
from tqdm import tqdm
from typing import List
import random

STEP_STARTIDX = -16
STEP_ENDIDX   = -4

cwd = os.getcwd()
TXT_PATH = os.path.join(cwd, "tx.txt")
output = []
with open(TXT_PATH, 'r') as fin:
    lines = fin.readlines()[STEP_STARTIDX:STEP_ENDIDX]
    print(lines)
    output.append(''.join(lines))
    print(output)