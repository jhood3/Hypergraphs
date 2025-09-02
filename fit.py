from utils import * 
import argparse
import numpy as np
import autograd.numpy as anp
from scipy.special import logsumexp
from math import isnan
from autograd import grad
import os
from scipy.stats import poisson
import argparse
import time
import matplotlib.pyplot as plt

print("Parsing command-line arguments...")
args = get_parsed_args([])
assert args.model in ["semi", "omni"], "model must be 'semi' or 'omni'"

print("\n--- Script Running with a Subset of Arguments ---")
print(f"Model          = {args.model}")
print(f"Directory      = {args.directory}")
print(f"Learning Rate  = {args.LEARNING_RATE}")
print(f"Test Mode      = {args.test}")
print(f"Random Seed    = {args.seed}")
print("-------------------------------------------------")

np.random.seed(args.seed)

if args.test:
    V, D, Y_indices_D, Y_counts_D, Y_indices_test_D, Y_counts_test_D, inds_VD = \
        load_data(args.directory, args.MAX_ORDER, args.MIN_ORDER, test=True)
    args.Y_indices_test_D = Y_indices_test_D
    args.Y_counts_test_D = Y_counts_test_D
else:
    V, D, Y_indices_D, Y_counts_D, inds_VD = \
        load_data(args.directory, args.MAX_ORDER, args.MIN_ORDER)

args.V, args.D, args.Y_indices_D, args.Y_counts_D, args.inds_VD = V, D, Y_indices_D, Y_counts_D, inds_VD

best_model = None
best_elbo = -float("inf")
start_time = time.time()

print("Beginning Training")
for m in range(args.NUM_RESTARTS):
    model = Model(args)
    for s in range(args.MAX_ITER):
        iter_time = time.time()
        model.update()
        if model.change_elbo < 1e-5:
            break

    if model.old_elbo > best_elbo:
        best_model = model
        best_elbo = model.old_elbo

print(best_elbo)
if args.test is True:
    best_model.test_stuff()
    print(best_model.heldout_llk_D)
print(f"Time is {time.time() - start_time}")

        
        
    


