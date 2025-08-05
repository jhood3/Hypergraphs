import numpy as np
import julia 
import subprocess
import json

def fit_omni(
    directory="supreme-court",
    test=False,
    C=3,
    K=6,
    model="semi",
    MIN_ORDER=2,
    MAX_ORDER=9,
    CONV_TOL=1.0,
    MAX_ITER=1000,
    LEARNING_RATE=1e-5,
    NUM_RESTARTS=10,
    NUM_STEPS=1,
    CHECK_EVERY=10,
    seed=123
):
    cmd = [
        "julia", "fit.jl",
        "--directory", directory,
        "--C", str(C),
        "--K", str(K),
        "--model", model,
        "--MIN_ORDER", str(MIN_ORDER),
        "--MAX_ORDER", str(MAX_ORDER),
        "--CONV_TOL", str(CONV_TOL),
        "--MAX_ITER", str(MAX_ITER),
        "--LEARNING_RATE", str(LEARNING_RATE),
        "--NUM_RESTARTS", str(NUM_RESTARTS),
        "--NUM_STEPS", str(NUM_STEPS),
        "--CHECK_EVERY", str(CHECK_EVERY),
        "--seed", str(seed)
    ]
    
    if test:
        cmd.append("--test")


    result = subprocess.run(cmd, capture_output=True, text=True)
    lines = result.stdout.strip().splitlines()
    json_line = None
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            json_line = line
            break
    
    output = json.loads(json_line)
    return output

    directory = "supreme-court"
test = True #if true, masks proportion of data as described in paper
C = 3
K = 6
model = "semi" #"semi" (faster) or "omni" (slower but more flexible) 
MIN_ORDER = 2 #set >= 2
MAX_ORDER = 9 #D in paper
CONV_TOL = 1 #convergence tolerance for change in log likelihood
MAX_ITER = 1000
LEARNING_RATE = 1e-5 #learning rate 
NUM_RESTARTS = 10 #number of random restarts
NUM_STEPS = 1 #steps of gradient ascent per iteration: set to 1 by default
CHECK_EVERY = 10
seed = 123