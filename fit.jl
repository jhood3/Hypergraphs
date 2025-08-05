include("utils.jl")

println("Parsing command-line arguments...")

args = get_parsed_args()

println("\n--- Script Running with a Subset of Arguments ---")
println("Model          = ", args["model"])
println("Directory      = ", args["directory"])
println("Learning Rate  = ", args["LEARNING_RATE"])
println("Test Mode      = ", args["test"])
println("Random Seed    = ", args["seed"])
println("-------------------------------------------------")

directory = args["directory"]
test = args["test"]
C = args["C"]
K = args["K"]
model = args["model"]
MIN_ORDER = args["MIN_ORDER"]
MAX_ORDER = args["MAX_ORDER"]
CONV_TOL = args["CONV_TOL"]
MAX_ITER = args["MAX_ITER"]
LEARNING_RATE = args["LEARNING_RATE"]
NUM_RESTARTS = args["NUM_RESTARTS"]
NUM_STEPS = args["NUM_STEPS"]
CHECK_EVERY = args["CHECK_EVERY"]
seed = args["seed"]
Random.seed!(seed) 

@assert model == "semi" || model == "omni" "model must be 'semi' or 'omni'"

###---------------------------------------------------LOAD DATA---------------------------------------------------

if test == true
    V, D, Y_indices_D, Y_counts_D, Y_indices_test_D, Y_counts_test_D, inds_VD = load_data(directory, MAX_ORDER, MIN_ORDER, true)
else
    V, D, Y_indices_D, Y_counts_D, inds_VD = load_data(directory, MAX_ORDER, MIN_ORDER)
end

###----------------------------------------------------------------------------------------------------------------


log_likelihoods = zeros(NUM_RESTARTS)
global start_time_global = time()
for i in 1:NUM_RESTARTS
    ###---------------------------------------------------INITIALIZATION---------------------------------------------------
    global old_elbo, change_elbo, s, times, heldout_llk_D = -1e10, 20000, 1, [], zeros(0, D - MIN_ORDER + 1)
    w_KC, init_W_KC, log_w_KC, gammas_DK, Theta_IC, Theta_IK, phi_VDK, phi_DK = init(K, C, D, V)
    println("Beginning training, random restart: ", i)

    ###---------------------------------------------------TRAINING---------------------------------------------------
    while (s <= MAX_ITER && change_elbo > CONV_TOL)
        start_time = time()
        if model =="omni"
            Y_CV, Y_KC, Y_DK = allocate_all_d(Y_indices_D, Y_counts_D, gammas_DK, Theta_IK, w_KC, Theta_IC, MIN_ORDER, D)
            gammas_DK = update_gamma_DK_d(Y_DK, phi_DK, MIN_ORDER, w_KC, 0, 0)
            Theta_IK, phi_VDK, phi_DK, Theta_IC = update_theta_all_d(Theta_IK, Theta_IC, phi_VDK, gammas_DK, phi_DK, Y_CV, w_KC, MIN_ORDER)
            if (K != C)
                log_w_KC, w_KC = optimize_Welbo_d(Y_KC, log_w_KC, Theta_IC, gammas_DK, MIN_ORDER, D, s, LEARNING_RATE, NUM_STEPS)
            end
        else
            Y_CV, Y_KC, Y_DK = allocate_all(Y_indices_D, Y_counts_D, gammas_DK, Theta_IK, w_KC, Theta_IC, MIN_ORDER, D)
            gammas_DK = update_gamma_DK(gammas_DK, Y_DK, phi_DK, 0, 0)
            Theta_IK, phi_VDK, phi_DK, Theta_IC = update_theta_all(Theta_IK, Theta_IC, phi_VDK, gammas_DK, phi_DK, Y_CV, w_KC, MIN_ORDER)
            if (K != C)
                log_w_KC, w_KC = optimize_Welbo(Y_KC, log_w_KC, Theta_IC, gammas_DK, MIN_ORDER, D, s, LEARNING_RATE, NUM_STEPS)
            end
        end

        Theta_IK, phi_DK, phi_VDK = update_params(Theta_IC, w_KC, D)

        end_time = time()
        push!(times, end_time - start_time)
        evaluate_convergence(s, old_elbo, Y_counts_D, Y_indices_D, gammas_DK, log_w_KC, Theta_IC, MIN_ORDER, model, CHECK_EVERY)
    end
    log_likelihoods[i] = likelihood
    if likelihood >= maximum(log_likelihoods[1:i][isnan.(log_likelihoods[1:i]).==false]) 
        global likelihood_best, w_KC_best, init_W_best, gammas_DK_best, Theta_IC_best, times_best, phi_DK_best = likelihood, w_KC, init_W_KC, gammas_DK, Theta_IC, times, phi_DK
        if test == true
            test_stuff(Y_indices_test_D, Y_counts_test_D, gammas_DK, Theta_IK, w_KC, MIN_ORDER, model)
            global heldout_llk_D_best = heldout_llk_D
        end
    end
end



# Collect the desired output
output = Dict(
    "w_KC" => w_KC_best,
    "Theta_IC" => Theta_IC_best,
    "gammas_DK" => gammas_DK_best,
)


println(JSON.json(output))

###---------------------------------------------------SAVING---------------------------------------------------
println("Total Time Elapsed: ", time() - start_time_global)
println("Best log-likelihood: ", likelihood_best)
save_params(test, w_KC_best, gammas_DK_best, Theta_IC_best, times_best, directory, model)




