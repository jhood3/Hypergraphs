include("helpers.jl")

directory = "workplace"
test = true
C = 5
K = 5
MIN_ORDER = 2
MAX_ORDER = 25
CONV_TOL = 1
MAX_ITER = 10
LEARNING_RATE = 1e-5
NUM_RESTARTS = 1
CHECK_EVERY = 10
seed = 101
Random.seed!(seed)


###---------------------------------------------------INITIALIZATION---------------------------------------------------

if test == true
    V, D, Y_indices_D, Y_counts_D, Y_indices_test_D, Y_counts_test_D, inds_VD = load_data(directory, MAX_ORDER, MIN_ORDER, true)
else
    V, D, Y_indices_D, Y_counts_D, inds_VD = load_data(directory, MAX_ORDER, MIN_ORDER)
end

log_likelihoods = zeros(NUM_RESTARTS)
global start_time_global = time()
for i in 1:NUM_RESTARTS
    global old_elbo = -1e10
    global change_elbo = 2
    global s = 1
    global times = []
    global llks = zeros(0, D - MIN_ORDER + 1)
    w_KC, log_w_KC, lambdas_DK, Theta_IC, Theta_IK, phi_VDK, phi_DK = init(K, C, D, V)
    init_W_KC = copy(w_KC)
    println("Beginning training, random restart: ", i)


    ###---------------------------------------------------TRAINING---------------------------------------------------
    while (s <= MAX_ITER && change_elbo > CONV_TOL)
        start_time = time()
        #allocate counts
        Y_CV, Y_KC, Y_DK = allocate_all(Y_indices_D, Y_counts_D, lambdas_DK, Theta_IK, w_KC, Theta_IC, MIN_ORDER, D)

        #update lambdas
        #lambdas_DK = update_lambda_DK(lambdas_DK, Y_DK, phi_DK, 1e-2, 1e-2)

        #Update community membership matrix Theta_IC
        #Theta_IK, phi_VDK, phi_DK, Theta_IC = update_theta_all(Theta_IK, Theta_IC, phi_VDK, lambdas_DK, phi_DK, Y_CV, w_KC, MIN_ORDER)

        #update w_KC. If K=C, then running Hypergraph-MT and do not update w_KC
        if K != C
            log_w_KC, w_KC =
                optimize_Welbo(Y_KC, log_w_KC, Theta_IC, lambdas_DK, MIN_ORDER, s, LEARNING_RATE, 100)

        end

        #update remaining parameters
        #Theta_IK, phi_DK, phi_VDK = update_params(Theta_IC, w_KC, D)

        #evaluate heldout log-likelihood and convergence
        end_time = time()
        push!(times, end_time - start_time)
        evaluate_convergence(s, old_elbo, Y_counts_D, Y_indices_D, lambdas_DK, log_w_KC, Theta_IC, MIN_ORDER, CHECK_EVERY)
    end
    log_likelihoods[i] = likelihood
    if likelihood >= maximum(log_likelihoods[1:i])
        global w_KC_best = w_KC
        global init_W_best = init_W_KC
        global lambdas_DK_best = lambdas_DK
        global Theta_IC_best = Theta_IC
        global times_best = times
        global phi_DK_best = phi_DK
        if test == true
            test_stuff(Y_indices_test_D, Y_counts_test_D, lambdas_DK, Theta_IK, MIN_ORDER)
            global llks_best = llks
        end
    end
end

###---------------------------------------------------SAVING---------------------------------------------------
println("Total Time Elapsed: ", time() - start_time_global)

if test == false
    w_KC_np = convert(Array{Float64}, w_KC)
    lambdas_DK_np = convert(Array{Float64}, lambdas_DK)
    Theta_IC_np = convert(Array{Float64}, Theta_IC)
    times_np = convert(Array{Float64}, times)
    npzwrite(
        "results/" * directory * "/C$(C)K$(K)seed$(seed)Params.npz",
        Dict("w_KC" => w_KC_np, "lambdas_DK" => lambdas_DK_np, "Theta_IC" => Theta_IC_np, "times" => times_np),
    )
else

end
