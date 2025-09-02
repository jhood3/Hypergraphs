# The utils file contains helper functions for data loading, training, 
# argument parsing, model saving, etc.
include("utils.jl")

println("Parsing command-line arguments...")

# parse arguments supplied to script from command line, see function in utils.jl for defaults
args = get_parsed_args()

# print a summary with subset of the parameters
println("\n--- Script Running with a Subset of Arguments ---")
println("Model          = ", args["model"])       # the type of model to run ('semi' or 'omni').
println("Directory      = ", args["directory"])   # the input directory that contains the hypergraph data
println("Learning Rate  = ", args["LEARNING_RATE"]) # the learning rate for the update to W
println("Test Mode      = ", args["test"])        # a boolean flag indicating if a portion of the data is heldout for evaluation. 
println("Random Seed    = ", args["seed"])        # the seed for the random number generator (for reproducibility)
println("-------------------------------------------------")

# --- Extract and assign script parameters from the 'args' dictionary to local variables ---
directory = args["directory"]      # Data and results directory.
test = args["test"]                # Boolean for test mode: True = test, False = train only
C = args["C"]                      # Number of classes 
K = args["K"]                      # Number of communities
model = args["model"]              # Model type ("semi" or "omni")
MIN_ORDER = args["MIN_ORDER"]      # Minimum hyperedge size/order (default is 2)
MAX_ORDER = args["MAX_ORDER"]      # Maximum hyperedge size/order (default is 25)
CONV_TOL = args["CONV_TOL"]        # Convergence tolerance for the change in ELBO.
MAX_ITER = args["MAX_ITER"]        # Maximum number of iterations for the training loop (if convergence tolerance is not reached before then)
LEARNING_RATE = args["LEARNING_RATE"] # Learning rate
NUM_RESTARTS = args["NUM_RESTARTS"] # Number of random restarts
NUM_STEPS = args["NUM_STEPS"]      # Number of gradient steps in update W step
CHECK_EVERY = args["CHECK_EVERY"]  # Check for convergence (compute ELBO) every CHECK_EVERY iterations
seed = args["seed"]                # Seed for random number generation

# Set global random seed 
Random.seed!(seed) 

# Assert that the model is one of the two supported types
@assert model == "semi" || model == "omni" "model must be 'semi' or 'omni'"

###---------------------------------------------------LOAD DATA---------------------------------------------------

# Load data. If in test mode, split data into test and train sets.
if test == true
    # If in test mode, load both training and testing datasets.
    V, D, Y_indices_D, Y_counts_D, Y_indices_test_D, Y_counts_test_D, inds_VD = load_data(directory, MAX_ORDER, MIN_ORDER, true)#, true)
else
    # If not in test mode, all data is training data.
    V, D, Y_indices_D, Y_counts_D, inds_VD = load_data(directory, MAX_ORDER, MIN_ORDER,false)#,true)
end

###----------------------------------------------------------------------------------------------------------------

# Initialize an array to store the final log-likelihood from each random restart.
log_likelihoods = zeros(NUM_RESTARTS)
# Record the start time of the entire script to measure total execution time.
global start_time_global = time()

# --- main training loop: includes multiple restarts ---
# This loop runs the entire training process multiple times to find the model with the highest log-likelihood
for i in 1:NUM_RESTARTS
    ###---------------------------------------------------INITIALIZATION---------------------------------------------------
    # Initialize variables for training run i out of NUM_RESTARTS.
    # 'old_elbo' stores the ELBO from the previous step to check for convergence.
    # 'change_elbo' is the change in ELBO between steps.
    # 's' is the iteration counter.
    # 'times' will store the duration of each iteration.
    global old_elbo, change_elbo, s, times, heldout_llk_D = -1e10, 20000, 1, [], zeros(0, D - MIN_ORDER + 1)
    w_KC, init_W_KC, log_w_KC, gammas_DK, Theta_IC, Theta_IK, phi_VDK, phi_DK = init(K, C, D, V)
    println("Beginning training, random restart: ", i)

    ###---------------------------------------------------TRAINING---------------------------------------------------
    # This is the core optimization loop for a single restart.
    # It continues until the maximum number of iterations is reached or the model converges.
    while (s <= MAX_ITER && change_elbo > CONV_TOL)
        start_time = time() 

        # The update steps vary by model type ('omni' or 'semi').
        if model =="omni"
            Y_CV, Y_KC, Y_DK = allocate_all_d(Y_indices_D, Y_counts_D, gammas_DK, Theta_IK, w_KC, Theta_IC, MIN_ORDER, D) #E step
            gammas_DK = update_gamma_DK_d(Y_DK, phi_DK, MIN_ORDER, w_KC, 0, 0) #update gamma
            Theta_IK, phi_VDK, phi_DK, Theta_IC = update_theta_all_d(Theta_IK, Theta_IC, phi_VDK, gammas_DK, phi_DK, Y_CV, w_KC, MIN_ORDER) #update node-class membership matrix Theta
            if (K != C) # Take gradient steps on free parameters of w_KC if K > C
                log_w_KC, w_KC = optimize_Welbo_d(Y_KC, log_w_KC, Theta_IC, gammas_DK, MIN_ORDER, D, s, LEARNING_RATE, NUM_STEPS)
            end
        else # 'semi' model
            Y_CV, Y_KC, Y_DK = allocate_all(Y_indices_D, Y_counts_D, gammas_DK, Theta_IK, w_KC, Theta_IC, MIN_ORDER, D) #E step
            gammas_DK = update_gamma_DK(gammas_DK, Y_DK, phi_DK, 0, 0) #update gamma
            Theta_IK, phi_VDK, phi_DK, Theta_IC = update_theta_all(Theta_IK, Theta_IC, phi_VDK, gammas_DK, phi_DK, Y_CV, w_KC, MIN_ORDER) #update node-class membership matrix Theta
            if (K != C) # Take gradient steps on free parameters of w_KC if K > C
                log_w_KC, w_KC = optimize_Welbo(Y_KC, log_w_KC, Theta_IC, gammas_DK, MIN_ORDER, D, s, LEARNING_RATE, NUM_STEPS)
            end
        end

        # Update parameters used for computation throughout that can be cached
        Theta_IK, phi_DK, phi_VDK = update_params(Theta_IC, w_KC, D)

        end_time = time() 
        push!(times, end_time - start_time) # store the duration of this iteration

        # Compute elbo, evaluate the model's convergence
        evaluate_convergence(s, old_elbo, Y_counts_D, Y_indices_D, gammas_DK, log_w_KC, Theta_IC, MIN_ORDER, model, CHECK_EVERY)
    end 
    # store the log-likelihood for the current restart
    log_likelihoods[i] = likelihood 

    # --- Check if the current model is the best one found so far ---
    if likelihood >= maximum(log_likelihoods[1:i][isnan.(log_likelihoods[1:i]).==false]) 
        # If this is the best model, save its parameters and performance metrics (if test is true)
        global likelihood_best, w_KC_best, init_W_best, gammas_DK_best, Theta_IC_best, times_best, phi_DK_best = likelihood, w_KC, init_W_KC, gammas_DK, Theta_IC, times, phi_DK
        
        if test == true
            global Theta_IK_best = Theta_IC_best * w_KC_best'
            test_stuff(Y_indices_test_D, Y_counts_test_D, gammas_DK_best, Theta_IK, w_KC_best, MIN_ORDER, model)
            global heldout_llk_D_best = heldout_llk_D
        end
    end
end

# --- Final Output and Saving ---

# Collect the desired final parameters from the best run into a dictionary and save them
output = Dict(
    "w_KC" => w_KC_best,
    "Theta_IC" => Theta_IC_best,
    "gammas_DK" => gammas_DK_best,
)

save_params(test, w_KC_best, gammas_DK_best, Theta_IC_best, times_best, directory, model, likelihood_best)



println("Total Time Elapsed: ", time() - start_time_global)
println("Best log-likelihood: ", likelihood_best)

