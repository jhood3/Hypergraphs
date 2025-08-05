using Statistics
using Distributions
using Statistics
using Base.Threads
using Random
using LinearAlgebra
using StatsFuns
using SpecialFunctions
using Combinatorics
using NPZ
using Base.Threads
using Distances
using Zygote
using ArgParse
using JSON


# Best Practice: The function is renamed to avoid name collision with the package function.
function get_parsed_args()
    s = ArgParseSettings(
        #description = "A script to run a model with various hyperparameters.",
        #epilog = "Example usage: julia your_script_name.jl --model cnn --LEARNING_RATE 1e-4"
    )

    # The '!' is a Julia convention for functions that modify their arguments (here, 's')
    @add_arg_table! s begin
        "--directory"
            help = "Path to the input data directory"
            arg_type = String
            default = "supreme-court"
        "--test"
            help = "If set, runs the script in test mode (a boolean flag)"
            action = :store_true # Standard way to handle flags. False by default.
        "--C"
            help = "Parameter C for the model"
            arg_type = Int
            default = 3
        "--K"
            help = "Parameter K for the model"
            arg_type = Int
            default = 6
        "--model"
            help = "Specify the model type (e.g., 'semi', 'cnn')"
            arg_type = String
            default = "semi"
        "--MIN_ORDER"
            help = "Minimum order for a model component"
            arg_type = Int
            default = 2
        "--MAX_ORDER"
            help = "Maximum order for a model component"
            arg_type = Int
            default = 9
        "--CONV_TOL"
            help = "Convergence tolerance for the optimization"
            arg_type = Float64
            default = 1.0
        "--MAX_ITER"
            help = "Maximum number of iterations"
            arg_type = Int
            default = 1000
        "--LEARNING_RATE"
            help = "Learning rate for the optimizer"
            arg_type = Float64
            default = 1e-5
        "--NUM_RESTARTS"
            help = "Number of random restarts for the optimization"
            arg_type = Int
            default = 10
        "--NUM_STEPS"
            help = "Number of steps to run"
            arg_type = Int
            default = 1
        "--CHECK_EVERY"
            help = "Frequency of checking for convergence (in iterations)"
            arg_type = Int
            default = 10
        "--seed"
            help = "Random seed for reproducibility"
            arg_type = Int
            default = 123
    end


    return parse_args(ARGS, s)
end




function save_params(test, w_KC_best, gammas_DK_best, Theta_IC_best, times_best, directory, model)
    if test == false
        w_KC_np = convert(Array{Float64}, w_KC_best)
        gammas_DK_np = convert(Array{Float64}, gammas_DK_best)
        Theta_IC_np = convert(Array{Float64}, Theta_IC_best)
        times_np = convert(Array{Float64}, times_best)
        mkpath("results/" * directory)
        if model == "semi"
        npzwrite(
            "results/" * directory * "/C$(C)K$(K)seed$(seed)Params.npz",
            Dict("w_KC" => w_KC_np, "gammas_DK" => gammas_DK_np, "Theta_IC" => Theta_IC_np, "times" => times_np),
        )
        elseif model == "omni"
            npzwrite(
                "results/" * directory * "/C$(C)K$(K)seed$(seed)ParamsD.npz",
                Dict("w_KC" => w_KC_np, "gammas_DK" => gammas_DK_np, "Theta_IC" => Theta_IC_np, "times" => times_np),
            )
        else
            println("Invalid model")
        end
    end
end

function softmax(x, dims)
    exps = exp.(x .- maximum(x, dims=dims))
    return exps ./ sum(exps, dims=dims)
end


function allocate_all_d(Y_indices_D, Y_counts_D, gammas_DK, Tau_IK, w_KC, Theta_IC, MIN_ORDER, D)
    C = size(w_KC, 2)
    K = size(w_KC, 1)
    V = size(Tau_IK, 1)
    Y_CV, Y_KC, Y_DK = reset_Y(D, V, C, K)
    locker = Threads.SpinLock()
    @threads for d = MIN_ORDER:D
        Y_indices = Y_indices_D[d]
        Y_counts = Y_counts_D[d]
        gammas_K = gammas_DK[d, :]
        inds_V = inds_VD[d]
        Y_IK, Y_IKC = allocate_K_d(Tau_IK, w_KC, Y_indices, Y_counts, gammas_K, Theta_IC)
        Y_CV_raw = dropdims(sum(Y_IKC, dims = 2), dims=2)'
        Y_KC_raw = dropdims(sum(Y_IKC, dims = 1), dims=1)
        lock(locker)
        Y_CV += Y_CV_raw
        Y_KC += Y_KC_raw
        Y_DK[d, :] = sum(Y_IK, dims = 1)
        unlock(locker)
    end
    return Y_CV, Y_KC, Y_DK
end

function allocate_K_d(Tau_IK, weights_KC, Y_indices, Y_counts, gammas_K, Theta_IC)
    I = size(Tau_IK, 1)
    K = size(Tau_IK, 2)
    C = size(weights_KC, 2)
    D = size(Y_indices, 2)
    Y_IK = zeros(size(Y_indices, 1), K)
    Tau_IK = max.(Tau_IK, 1e-300)
    for d in axes(Y_indices, 2)
        Y_IK .+= log.(Tau_IK[Y_indices[:, d], :])
    end
    Y_IK = exp.(Y_IK) 
    @assert minimum(Y_IK) >= 0
    w_KC_d = weights_KC.^D
    Y_IC = Y_IK[:, 1:C]
    sub_IK = Y_IC * w_KC_d'
    Y_IK[:, (C+1):K] .-= sub_IK[:, (C+1):K]
    Y_IK .= max.(Y_IK, 1e-300)
    Y_IK .*= gammas_K'
    Y_IK ./= sum(Y_IK, dims = 2)
    Y_IK .*= Y_counts
    Y_IKC = zeros(I, K, C)
    w_KC_d = weights_KC .^ (D - 1)
    l_w_kc_d = log.(w_KC_d)
    l_w_kc = log.(weights_KC)
    @views @inbounds for i in eachindex(Y_counts)
        Y_IKC[Y_indices[i,:],:,:] .+= allocate_iCK(Tau_IK, weights_KC, Theta_IC, Y_IK[i,:], Y_indices[i,:], w_KC_d, l_w_kc_d, l_w_kc)
    end
    @assert minimum(Y_IK) >= 0 println(minimum(Y_IK))
    @assert sum(Y_IK) ≈ sum(Y_counts)
    @assert sum(Y_IKC) ≈ sum(Y_counts)*D
    return (Y_IK, Y_IKC)
end

function allocate_iCK(Tau_IK, weights_KC, Theta_IC, count_K, index, w_KC_d, l_w_kc_d, l_w_kc)
    D = length(index)
    C  = size(Theta_IC, 2)
    K = size(Tau_IK, 2)
    y_dKC = zeros(D, K, C)
    @inbounds for c in 1:C
        y_dKC[:, c, c] .= count_K[c]
    end
    if sum(count_K[(C+1):K]) > 1e-15 #otherwise not worth computational cost
    @views @inbounds for k in (C+1):K
        if count_K[k] > 1e-15 
            w_C = weights_KC[k, :]
            l_w_c = l_w_kc[k, :]
            w_C_d = w_KC_d[k, :]
            l_w_c_d = l_w_kc_d[k, :]
            @views @fastmath for d in 1:D
                potentials_C = l_w_c .+ log.(Theta_IC[index[d], :])
                constant = 0
                constant_C = l_w_c_d .+ 0
                @views for m in 1:D
                    if m != d
                        constant += log.(Tau_IK[index[m], k])
                        constant_C .+= log.(Theta_IC[index[m], :])
                    end
                end
                potentials_C .+= logsubexp.(constant, constant_C)
                y_dKC[d, k, :] = exp.(potentials_C .- logsumexp(potentials_C)) .* count_K[k]
            end
        end
    end
    end
    return(y_dKC)
end

function update_theta_iC_d(i, Y_iC, gamma_DK, weights_KC, start, D, phi_vDK, beta)
    numerator_C = Y_iC
    phi_vDC = phi_vDK[:, 1:C]
    d_val_sum = sum(gamma_DK[start:D, :] .* phi_vDK[(start-1):(D-1),:], dims=1)
    denominators_C = (d_val_sum * weights_KC)'
    
    @views for d in start:D
        term_KC = gamma_DK[d, :] .* weights_KC
        w_KC_d = weights_KC .^ (d-1)
        wp_KC = w_KC_d .* phi_vDC[d-1, :]'
        term_KC .*= wp_KC
        denominators_C .-= dropdims(sum(term_KC[(C+1):K, :], dims = 1), dims=1)
    end
    denominators_C .+= beta
    Theta_I = numerator_C ./ denominators_C
    return(Theta_I)
end

function update_theta_iC_D_d(
    i,
    Tau_IK,
    Theta_IC,
    phi_VDK,
    gammas_DK,
    phi_DK,
    Y_C,
    weights_KC,
    next_ind,
    start,
    alpha,
    beta,
    iter,
)
    old_theta = Tau_IK[i, :]
    new_theta_iC = update_theta_iC_d(i, Y_C .+ alpha, gammas_DK, weights_KC, start, size(phi_DK, 1), exp.(phi_VDK[i,:,:]), beta)
    new_theta_iC = max.(new_theta_iC, alpha)
    # Update theta values
    new_Tau_IK = weights_KC * new_theta_iC
    Tau_IK[i, :] .= new_Tau_IK
    Theta_IC[i, :] .= new_theta_iC
    phi_DK = update_phi_DK(old_theta, new_Tau_IK, phi_DK, i, phi_VDK)
    phi_VDK = update_phi_VDK(Tau_IK[next_ind, :], phi_DK, phi_VDK, next_ind)
    return (Tau_IK, phi_VDK, phi_DK, Theta_IC)
end


function update_theta_all_d(Tau_IK, Theta_IC, phi_VDK, gammas_DK, phi_DK, Y_CV, w_KC, MIN_ORDER)
    V = size(Tau_IK, 1)
    @views for i = 1:V
        j = min(i+1, V)
        Tau_IK, phi_VDK, phi_DK, Theta_IC = update_theta_iC_D_d(
            i,
            Tau_IK,
            Theta_IC,
            phi_VDK,
            gammas_DK,
            phi_DK,
            Y_CV[:, i],
            w_KC,
            j,
            MIN_ORDER,
            0,
            0,
            0
        )
    end
    Theta_IC = Theta_IC ./ sum(Theta_IC) .* V#/C
    Theta_IK = Theta_IC * w_KC'
    return(Tau_IK, phi_VDK, phi_DK, Theta_IC)
end

function compute_gradient_d(log_w_KC, Y_KC, Theta_IC, gammas_DK, start, D)
    return Zygote.gradient(
        w -> compute_Welbo_d(Y_KC, w, Theta_IC, gammas_DK[start:D,:], start, D),
        log_w_KC
    )[1]
end

function optimize_Welbo_d(
    Y_KC,
    log_w_KC,
    Theta_IC,
    gammas_DK,
    start,
    D,
    s,
    lr = 1e-5,
    steps = 1,
)
grad_w_KC = zeros(size(log_w_KC))
for step = 1:steps
    grad_w_KC .= compute_gradient_d(log_w_KC, Y_KC, Theta_IC, gammas_DK, start, D)
    @inbounds grad_w_KC = clamp.(grad_w_KC, -1e5, 1e5)
    @inbounds grad_w_KC[isnan.(grad_w_KC)] .= 0
    log_w_KC[(C+1):K,:] .+= lr * grad_w_KC[(C+1):K,:]
end
    return log_w_KC, softmax(log_w_KC, 2)#log.(exp.(log_w_KC) .+ 1) 
end

function compute_Welbo_d(Y_KC, log_w_KC, Theta_IC, gammas_DK, start, D)
    #w_KC = log1p.(exp.(log_w_KC))  # log(1 + exp(x)) is equivalent
    w_KC = softmax(log_w_KC, 2) #under softmax formulation
    phi_DK = compute_phi_DK_min(Theta_IC * w_KC', D)
    phi_DK_sub = phi_DK[start:D, :]
    llk = sum(Y_KC .* log.(w_KC))
    llk -= sum(phi_DK_sub .* gammas_DK)
    for d in 1:(D-start + 1) #problem; re-evaluate
        llk += sum(gammas_DK[d,(C+1):K] .* dropdims(sum(w_KC[(C+1):K, :].^(d+ start-1) .* phi_DK_sub[d, 1:C]', dims=2), dims=2))
    end
    return llk
end


function compute_llk_d(Y_counts_D, Y_indices_D, gammas_DK, log_w_KC, Theta_IC, start)
    C = size(Theta_IC, 2)
    #w_KC = log1p.(exp.(log_w_KC))
    w_KC = softmax(log_w_KC, 2)
    Tau_IK = Theta_IC * w_KC'
    llk = 0
    phi_DK = compute_phi_DK_min(Tau_IK, D)
    llk -= sum(phi_DK[start:D, :] .* gammas_DK[start:D, :])
    for d in start:D
        llk += sum(gammas_DK[d,(C+1):K] .* dropdims(sum(w_KC[(C+1):K, :].^d .* phi_DK[d, 1:C]', dims=2), dims=2))
    end
    for d = start:D
        Y_counts = Y_counts_D[d]
        Y_indices = Y_indices_D[d]
        rates_IK = zeros(size(Y_indices, 1), K) 
        for m = 1:size(Y_indices, 2)
            rates_IK .+= log.(Tau_IK[Y_indices[:, m], :])# + rates_IK
        end
        rates_IK .= exp.(rates_IK)
        for k in (C+1):K
            rates_IK[:, k] .-= dropdims(sum(rates_IK[:, 1:C] .* w_KC[k, :]'.^d, dims=2), dims=2)
        end
        rates_IK .= max.(rates_IK, 0)
        rates_IK .= log.(rates_IK.+1e-80) .+ log.(gammas_DK[d, :])'
        llk += sum(logsumexp(rates_IK, dims = 2) .* Y_counts)
    end
    println("Log-likelihood: $(llk)")
    return llk
end

function make_predictions_d(Y_indices_test_D, Y_counts_test_D, gammas_DK, Tau_IK, start, w_KC)
    D = length(Y_indices_test_D)
    predictions = zeros(0)
    log_pmf = zeros(0)
    prob_greater_0 = zeros(0)
    first_half = zeros(0)
    second_half = zeros(0)
    log_pmfs = zeros(D - start + 1)
    aucs = zeros(D - start + 1)
    for d = start:D
        Y_indices = Y_indices_test_D[d]
        Y_counts = Y_counts_test_D[d]
        rates_IK = zeros(size(Y_indices, 1), K)
        for m = 1:d
            rates_IK += log.(Tau_IK[Y_indices[:, m], :])
        end
        rates_IK[isnan.(rates_IK)] .= -Inf
        rates_IK = exp.(rates_IK)
        w_KC_d = w_KC .^ (d)
        for k in (C+1):K
            diff = dropdims(sum(w_KC_d[k, :]' .* rates_IK[:, 1:C], dims = 2), dims = 2)
            rates_IK[:, k] .-= diff
        end
        rates_IK .*= gammas_DK[d, :]'
        predictions_d = sum(rates_IK, dims = 2)
        predictions_d = max.(predictions_d, 0)
        prob_greater_0_d = 1 .- exp.(-predictions_d)
        first_half_d = predictions_d[1:Int(round(length(prob_greater_0_d) / 2))]
        second_half_d = predictions_d[Int(round(length(prob_greater_0_d) / 2))+1:end]
        log_poisson = logpdf.(Poisson.(predictions_d .+ 1e-300), Y_counts)
        predictions = vcat(predictions, predictions_d)
        log_pmfs[d-start+1] = mean(log_poisson)
        log_pmf = vcat(log_pmf, log_poisson)
        prob_greater_0 = vcat(prob_greater_0, prob_greater_0_d)
        first_half = vcat(first_half, first_half_d)
        second_half = vcat(second_half, second_half_d)
        aucs[d-start+1] = sum(first_half_d .> second_half_d) / length(first_half_d)
    end
    auc =
        (sum(first_half .> second_half) + 0.5 * sum(first_half .== second_half)) /
        length(first_half)
    return log_pmfs, aucs
end

function update_theta_all(Theta_IK, Theta_IC, phi_VDK, gammas_DK, phi_DK, Y_CV, w_KC, MIN_ORDER)
    V = size(Theta_IK, 1)
    @views for i = 1:V
        j = min(i+1, V)
        Theta_IK, phi_VDK, phi_DK, Theta_IC = update_theta_iC_D(
            i,
            Theta_IK,
            Theta_IC,
            phi_VDK,
            gammas_DK,
            phi_DK,
            Y_CV[:, i],
            w_KC,
            j,
            MIN_ORDER,
            0,
            0,
            0,
        )
    end
    Theta_IC = Theta_IC ./ sum(Theta_IC) .* V#/C
    Theta_IK = Theta_IC * w_KC'
    return(Theta_IK, phi_VDK, phi_DK, Theta_IC)
end

function test_stuff(Y_indices_test_D, Y_counts_test_D, gammas_DK, Theta_IK, w_KC, MIN_ORDER, model)
    if model == "omni"
        log_pmf, auc = make_predictions_d(Y_indices_test_D, Y_counts_test_D, gammas_DK, Theta_IK, MIN_ORDER, w_KC)
    elseif model == "semi"
        log_pmf, auc = make_predictions(Y_indices_test_D, Y_counts_test_D, gammas_DK, Theta_IK, MIN_ORDER)
    else
        println("Invalid model")
    end
    global heldout_llk_D = vcat(heldout_llk_D, log_pmf')
    times_np = convert(Array{Float64}, times)
    heldout_llk_D_np = convert(Array{Float64}, heldout_llk_D)
    if model == "omni"
        npzwrite(
            "results/" * directory * "/C$(C)K$(K)seed$(seed)RRD.npz",
            Dict("times" => times_np, "llks" => heldout_llk_D_np),
        )
    elseif model == "semi"
        npzwrite(
            "results/" * directory * "/C$(C)K$(K)seed$(seed)RR.npz",
            Dict("times" => times_np, "llks" => heldout_llk_D_np),
        )
    else
        println("Invalid model")
    end
end

function evaluate_convergence(s, old_elbo, Y_counts_D, Y_indices_D, gammas_DK, log_w_KC, Theta_IC, MIN_ORDER, model, CHECK_EVERY = 10)
    if s % CHECK_EVERY == 0
        println("Iteration: ", s)
        if model == "omni"
            global likelihood = compute_llk_d(Y_counts_D, Y_indices_D, gammas_DK, log_w_KC, Theta_IC, MIN_ORDER)
        elseif model == "semi"
            global likelihood = compute_llk(Y_counts_D, Y_indices_D, gammas_DK, log_w_KC, Theta_IC, MIN_ORDER)
        else
            println("Invalid model")
        end
        global change_elbo = abs(likelihood - old_elbo)
        global old_elbo = likelihood
    end
    global s += 1
end



function reset_Y(D, V, C, K)
    Y_CV = zeros(C,V)
    Y_KC = zeros(K, C)
    Y_DK = zeros(D, K)
    return Y_CV, Y_KC, Y_DK
end

function init(K, C, D, V)
    w_KC, log_w_KC = init_W(K, C) 
    init_W_KC = copy(w_KC)
    log_w_KC = log.(w_KC)
    gammas_DK = ones(D, K)
    Theta_IC, Theta_IK = init_membership(V, C, K, D, w_KC)
    Theta_IC = Theta_IC ./ sum(Theta_IC) .* V/C
    Theta_IK, phi_DK, phi_VDK = update_params(Theta_IC, w_KC, D)
    return w_KC, init_W_KC, log_w_KC, gammas_DK, Theta_IC, Theta_IK, phi_VDK, phi_DK
end

function load_data(directory, MAX_ORDER = 25, MIN_ORDER = 2, test = false)
    #data = npzread("data/" * directory * "/edges$(MIN_ORDER).npz")
    #data = npzread("data/" * directory * "/edges3ce.npz")
    data = npzread("data/" * directory * "/edges.npz")
    Y_indices_raw = data["edges"]
    Y_counts_raw = round.(data["counts"])
    D = minimum([size(Y_indices_raw, 2), MAX_ORDER])
    D_uncapped = size(Y_indices_raw, 2)
    if D_uncapped > MAX_ORDER
        Y_indices_raw = Y_indices_raw[:, 1:(D+1)]
    end
    Y_indices_D = Array{Array{Int,2}}(undef, D)
    Y_counts_D = Array{Array{Int,1}}(undef, D)
    inds_VD = Array{Array{Array{Int,1},1}}(undef, D)
    for i = 1:MIN_ORDER-1
        Y_indices_D[i] = zeros(1, 1)
        Y_counts_D[i] = zeros(1)
    end

    V = Int(maximum(Y_indices_raw))
    for d = MIN_ORDER:D
        if D_uncapped > MAX_ORDER
            indices = findall(x -> sum(x .== 0) == D - d + 1, eachrow(Y_indices_raw))
        else
            indices = findall(x -> sum(x .== 0) == D - d, eachrow(Y_indices_raw))
        end
        Y_indices_D[d] = Y_indices_raw[indices, 1:d]
        Y_counts_D[d] = Y_counts_raw[indices]
    end

    if test == true
        Y_indices_test_D, Y_counts_test_D, Y_indices_train_D, Y_counts_train_D =
            holdout(Y_indices_D, Y_counts_D, MIN_ORDER, V)
        Y_indices_D = Y_indices_train_D
        Y_counts_D = Y_counts_train_D
    end


    for d = MIN_ORDER:D
        Y_indices = Y_indices_D[d]
        inds_V = []
        for i = 1:V
            push!(inds_V, find_matching_rows(Y_indices, i))
        end
        inds_VD[d] = inds_V
    end
    if test == true
        return V, D, Y_indices_D, Y_counts_D, Y_indices_test_D, Y_counts_test_D, inds_VD
    else
        return V, D, Y_indices_D, Y_counts_D, inds_VD
    end
end


function update_params(Theta_IC, w_KC, D)
    Theta_IK = Theta_IC * w_KC'
    phi_DK, phi_VDK = compute_phi_DK(Theta_IK, D)
    phi_DK = log.(phi_DK)
    phi_VDK = log.(phi_VDK)
    return Theta_IK, phi_DK, phi_VDK
end


function allocate_all(Y_indices_D, Y_counts_D, gammas_DK, Theta_IK, w_KC, Theta_IC, MIN_ORDER, D)
    C = size(w_KC, 2)
    K = size(w_KC, 1)
    V = size(Theta_IK, 1)
    Y_CV, Y_KC, Y_DK = reset_Y(D, V, C, K)
    locker = Threads.SpinLock()
    for d = MIN_ORDER:D
        Y_indices = Y_indices_D[d]
        Y_counts = Y_counts_D[d]
        gammas_K = gammas_DK[d, :]
        inds_V = inds_VD[d]
        Y_IK = allocate(Y_indices, Y_counts, gammas_K, Theta_IK)
        Y_CV_raw, Y_KC_raw = reallocate(Y_IK', inds_V, w_KC, Theta_IC)
        lock(locker)
        Y_CV += Y_CV_raw
        Y_KC += Y_KC_raw
        Y_DK[d, :] = sum(Y_IK, dims = 1)
        unlock(locker)
    end
    return Y_CV, Y_KC, Y_DK
end


function init_W(K, C)
    w_KC = zeros(K, C)
    for c = 1:C
        w_KC[c, c] += 1
    end
    for k = 1:K
        if k > C
            c = rand(1:C)
            c2 = rand(1:C)
            alpha = ones(C)./(k - C).*C
            w_KC[k, :] = rand(Dirichlet(alpha))
        end
    end
    return w_KC, log.(exp.(w_KC) .- 1)
end


function allocate(Y_indices, Y_counts, gammas_K, Theta_IK)
    I = size(Theta_IK, 1)
    K = size(Theta_IK, 2)
    Y_IK = zeros(size(Y_indices, 1), K) .+ log.(gammas_K)'
    for d in axes(Y_indices, 2)
        Y_IK += log.(Theta_IK[Y_indices[:, d], :])
    end
    Y_IK = exp.(Y_IK .- logsumexp(Y_IK, dims = 2)) .* (Y_counts)
    @assert sum(Y_IK) ≈ sum(Y_counts) println(sum(Y_IK), sum(Y_counts))
    return (Y_IK)
end

function gradient(Y_KC, Theta_IC, gammas_DK, start, log_w_KC)
    Zygote.gradient(
        w -> compute_Welbo(Y_KC, w, Theta_IC, gammas_DK[start:D,:], start),
        log_w_KC,
    )[1]
end





function compute_gradient(log_w_KC, Y_KC, Theta_IC, gammas_DK, start, D)
    return Zygote.gradient(
        w -> compute_Welbo(Y_KC, w, Theta_IC, gammas_DK[start:D,:], start, D),
        log_w_KC
    )[1]
end



function optimize_Welbo(
    Y_KC,
    log_w_KC,
    Theta_IC,
    gammas_DK,
    start,
    D,
    s,
    lr = 1e-5,
    steps = 1,
)
grad_w_KC = zeros(size(log_w_KC))
for step = 1:steps
    grad_w_KC .= compute_gradient(log_w_KC, Y_KC, Theta_IC, gammas_DK, start, D)
    @inbounds grad_w_KC = clamp.(grad_w_KC, -1e5, 1e5)
    @inbounds grad_w_KC[isnan.(grad_w_KC)] .= 0
    log_w_KC .+= lr * grad_w_KC
end
    return log_w_KC, softmax(log_w_KC, 2) #log.(exp.(log_w_KC) .+ 1)  #
end



function compute_Welbo(Y_KC, log_w_KC, Theta_IC, gammas_DK, start, D)
    w_KC = softmax(log_w_KC, 2)#log1p.(exp.(log_w_KC))  # log(1 + exp(x)) is equivalent
    phi_DK = compute_phi_DK_min(Theta_IC * w_KC', D)
    phi_DK_sub = phi_DK[start:D, :]
    llk = sum(Y_KC .* log.(w_KC))
    llk -= sum(phi_DK_sub .* gammas_DK)
    return llk
end

function compute_phi_DK(input_mat::Array{Float64}, D::Int)
    phi_DK = sum(input_mat, dims = 1)'
    phi_VK = phi_DK' .- input_mat
    phi_VDK = zeros(size(input_mat, 1), D, size(input_mat, 2))
    phi_VDK[:, 1, :] = phi_VK
    for d = 2:D
        new_phi_DK_row = sum(phi_VK .* input_mat, dims = 1)' / d
        if d == 2
            phi_DK = vcat(phi_DK', new_phi_DK_row')
        else
            phi_DK = vcat(phi_DK[1:(d-1), :], new_phi_DK_row')  # Create new array for phi_DK
        end
        phi_VK = new_phi_DK_row' .- phi_VK .* input_mat
        phi_VDK[:, d, :] = phi_VK
    end
    phi_DK = max.(phi_DK, 1e-10)
    phi_VDK = max.(phi_VDK, 1e-10)
    return phi_DK, phi_VDK
end




function compute_phi_DK_min(input_mat::Array{Float64}, D::Int)
    phi_DK = sum(input_mat, dims = 1)'
    phi_VK = phi_DK' .- input_mat
    for d = 2:D
        new_phi_DK_row = sum(phi_VK .* input_mat, dims = 1)' / d
        if d == 2
            phi_DK = vcat(phi_DK', new_phi_DK_row')
        else
            phi_DK = vcat(phi_DK[1:(d-1), :], new_phi_DK_row')  # Create new array for phi_DK
        end
        phi_VK = new_phi_DK_row' .- phi_VK .* input_mat
    end
    phi_DK = max.(phi_DK, 0.0)
    return phi_DK
end



function compute_llk(Y_counts_D, Y_indices_D, gammas_DK, log_w_KC, Theta_IC, start)
    w_KC = softmax(log_w_KC, 2)#log1p.(exp.(log_w_KC))
    Theta_IK = Theta_IC * w_KC'
    llk = 0
    phi_DK = compute_phi_DK_min(Theta_IK, D)
    llk -= sum(phi_DK[start:D, :] .* gammas_DK[start:D, :])
    for d = start:D
        Y_counts = Y_counts_D[d]
        Y_indices = Y_indices_D[d]
        rates_IK = zeros(size(Y_indices, 1), K) .+ log.(gammas_DK[d, :])'
        for m = 1:size(Y_indices, 2)
            rates_IK .+= log.(Theta_IK[Y_indices[:, m], :])# + rates_IK
        end
        llk += sum(logsumexp(rates_IK, dims = 2) .* Y_counts)
    end
    println("Log-likelihood: $(llk)")
    return llk
end


function reallocate(Y_KI, inds_V, weights_KC, Theta_IC)
    V = length(inds_V)
    K = size(Y_KI, 1)
    C = size(Theta_IC, 2)
    Y_CV = zeros(C, V)
    Y_KC = zeros(K, C)
    for i = 1:V
        yi = Y_KI[:, inds_V[i]]
        y_ik = dropdims(sum(yi, dims = 2), dims = 2)
        weight_KC = weights_KC .* Theta_IC[i, :]'
        col_sums = sum(weight_KC, dims = 2)
        col_sums[col_sums.==0] .= 1e-100 #stability
        weight_KC ./= col_sums
        Y_CV[:, i] .= transpose(weight_KC) * y_ik
        Y_KC .+= sum(yi, dims = 2) .* weight_KC
    end
    return (Y_CV, Y_KC)
end

function init_membership(V, C, K, D, weights_KC)
    Theta_IC = zeros(V, C)
    Theta_IK = zeros(V, K)
    phi_DK = zeros(D, K)
    phi_VDK = zeros(V, D, K)
    u_C = rand(Dirichlet(ones(C) .* 30))
    u_K = weights_KC * u_C

    for d = 1:D
        for k = 1:K
            phi_DK[d, k] =
                lgamma(V + 1) - lgamma(V - d + 1) - lgamma(d + 1) + d * log(u_K[k])
        end
    end

    for i = 1:V
        Theta_IC[i, :] .= u_C
        Theta_IK[i, :] .= u_K
    end

    for i = 1:V
        for d = 1:D
            if d >= 2
                phi_VDK[i, d, :] =
                    logsubexp.(phi_DK[d, :], log.(Theta_IK[i, :]) .+ phi_VDK[i, d-1, :])
            else
                @assert d == 1
                phi_VDK[i, d, :] = logsubexp.(phi_DK[d, :], log.(Theta_IK[i, :]))
            end
        end
    end
    for i = 1:V
        Theta_IK, Theta_IC =
            update_theta_iC_init(i, Theta_IK, Theta_IC, weights_KC)
    end
    return (Theta_IC, Theta_IK)
end



function update_theta_iC_D(
    i,
    Theta_IK,
    Theta_IC,
    phi_VDK,
    gammas_DK,
    phi_DK,
    Y_C,
    weights_KC,
    next_ind,
    start,
    alpha,
    beta,
    iter,
)
    old_theta = Theta_IK[i, :]
    
    # Precompute the exponential part to avoid repeated computation
    exp_phi_VDK = exp.(phi_VDK[i, (start-1):(size(phi_DK, 1)-1), :])

    # Efficient sum computation of d_val
    d_val_sum = sum(gammas_DK[start:size(phi_DK, 1), :] .* exp_phi_VDK, dims=1)

    # Compute denominators and new_theta_iC
    denominators = d_val_sum * weights_KC
    new_theta_iC = (Y_C .+ alpha) ./ (denominators' .+ beta)  # shrinkage prior
    # Update theta values
    new_theta_iK = weights_KC * new_theta_iC
    Theta_IK[i, :] .= new_theta_iK
    Theta_IC[i, :] .= new_theta_iC

    # Update phi_DK and phi_VDK
    phi_DK = update_phi_DK(old_theta, new_theta_iK, phi_DK, i, phi_VDK)
    phi_VDK = update_phi_VDK(Theta_IK[next_ind, :], phi_DK, phi_VDK, next_ind)

    return (Theta_IK, phi_VDK, phi_DK, Theta_IC)
end


function holdout(Y_indices_D, Y_counts_D, start, V)
    Y_indices_test_D = []
    Y_counts_test_D = []
    Y_indices_train_D = []
    Y_counts_train_D = []
    D = length(Y_indices_D)
    for d = 1:(start-1)
        push!(Y_indices_train_D, 0)
        push!(Y_counts_train_D, 0)
        push!(Y_indices_test_D, 0)
        push!(Y_counts_test_D, 0)
    end
    for d = start:D
        println(d)
        Y_indices = Y_indices_D[d]
        Y_counts = Y_counts_D[d]
        n = size(Y_indices, 1)
        n_test = Int(floor(n * 0.1))
        n_test = min(n_test, 1000)
        if n_test > 0
            test_indices = rand(1:n, n_test)
            train_indices = setdiff(1:n, test_indices)
        else
            test_indices = []
            train_indices = 1:n
        end

        test_indices_zero = zeros(n_test, d)
        vector_of_tuples = [Tuple(Y_indices[i, :]) for i = 1:size(Y_indices, 1)]

        for i = 1:n_test #sample combinations of nodes from 1:V that do not appear in Y_indices or test_indices_zero
            new = sample(1:V, d, replace = false)
            while Tuple(new) in vector_of_tuples
                new = sample(1:V, d, replace = false)
            end
            push!(vector_of_tuples, Tuple(new))
            test_indices_zero[i, :] .= new
        end
        Y_indices_test = Int.(vcat(Y_indices[test_indices, :], test_indices_zero))
        Y_counts_test = vcat(Y_counts[test_indices], zeros(n_test))
        push!(Y_indices_test_D, Y_indices_test)
        push!(Y_counts_test_D, Y_counts_test)
        push!(Y_indices_train_D, Y_indices[train_indices, :])
        push!(Y_counts_train_D, Y_counts[train_indices])
    end
    return Y_indices_test_D, Y_counts_test_D, Y_indices_train_D, Y_counts_train_D
end

function make_predictions(Y_indices_test_D, Y_counts_test_D, gammas_DK, Theta_IK, start)
    D = length(Y_indices_D)
    predictions = zeros(0)
    log_pmf = zeros(0)
    prob_greater_0 = zeros(0)
    first_half = zeros(0)
    second_half = zeros(0)
    log_pmfs = zeros(D - start + 1)
    aucs = zeros(D - start + 1)
    for d = start:D
        Y_indices = Y_indices_test_D[d]
        Y_counts = Y_counts_test_D[d]
        rates_IK = zeros(size(Y_indices, 1), K) .+ log.(gammas_DK[d, :])'
        for m = 1:d
            rates_IK += log.(Theta_IK[Y_indices[:, m], :])
        end
        #if length(Y_counts) > 0
        #println(maximum(exp.(rates_IK)))
        #end
        #if nan, set to -Inf
        rates_IK[isnan.(rates_IK)] .= -Inf
        predictions_d = sum(exp.(rates_IK), dims = 2)
        prob_greater_0_d = 1 .- exp.(-predictions_d)
        #println(predictions_d)
        #get first half of vector of prob_greater_0_d
        first_half_d = predictions_d[1:Int(round(length(prob_greater_0_d) / 2))]
        second_half_d = predictions_d[Int(round(length(prob_greater_0_d) / 2))+1:end]
        #auc = (sum(first_half > second_half) + 0.5*sum(first_half .== second_half)) / length(prob_greater_0_d)*2
        log_poisson = logpdf.(Poisson.(predictions_d .+ 1e-300), Y_counts)
        predictions = vcat(predictions, predictions_d)
        log_pmfs[d-start+1] = mean(log_poisson)
        aucs[d-start+1] = (sum(first_half_d .> second_half_d) + 0.5 * sum(first_half_d .== second_half_d))/length(first_half_d)
        log_pmf = vcat(log_pmf, log_poisson)
        prob_greater_0 = vcat(prob_greater_0, prob_greater_0_d)
        first_half = vcat(first_half, first_half_d)
        second_half = vcat(second_half, second_half_d)
    end
    auc =
        (sum(first_half .> second_half) + 0.5 * sum(first_half .== second_half)) /
        length(first_half)
    return log_pmfs, aucs
end



function update_phi_DK(old, new, phi_DK, i, phi_VDK)
    phi_DK[1, :] .= logsubexp.(phi_DK[1, :], log.(old))
    phi_DK[1, :] .= logsumexp.(phi_DK[1, :], log.(new))
    @views for d = 2:D
        phi_DK[d, :] .= logsumexp.(phi_DK[d, :], phi_VDK[i, d-1, :] .+ log.(new))
        phi_DK[d, :] .= logsubexp.(phi_DK[d, :], phi_VDK[i, d-1, :] .+ log.(old))
    end
    return (phi_DK)
end

function update_phi_VDK(theta_k, phi_DK, phi_VDK, j)
    phi_VDK[j, 1, :] .= logsubexp.(phi_DK[1, :], log.(theta_k))
    @views for d = 2:D
        phi_VDK[j, d, :] .= logsubexp.(phi_DK[d, :], log.(theta_k) .+ phi_VDK[j, d-1, :])
    end
    return (phi_VDK)
end


function update_theta_iC_init(i, Theta_IK, Theta_IC, weights_KC)
    C = size(Theta_IC, 2)
    new_theta_iC = rand(Dirichlet(ones(C)))
    new_theta_iK = weights_KC * new_theta_iC
    Theta_IK[i, :] .= new_theta_iK
    Theta_IC[i, :] .= new_theta_iC
    return (Theta_IK, Theta_IC)
end



function find_matching_rows(Y_IK, i_prime)
    matching_rows = [any(x -> x == i_prime, row) for row in eachrow(Y_IK)]
    true_vals = findall(matching_rows)
    return true_vals
end

function update_gamma_DK(gammas_DK, y_DK, phi_DK, alpha = 1e-2, beta = 1e-2)
    gammas_DK = (y_DK .+ alpha) ./ (exp.(phi_DK) .+ beta)
    return (gammas_DK)
end

function update_gamma_DK_d(Y_DK, phi_DK, start, weights_KC, alpha, beta)
    D = size(phi_DK, 1)
    K = size(phi_DK, 2)
    e_phi_DK = exp.(phi_DK)
    gammas_DK = ones(D, K)
    C = size(weights_KC, 2)
    for d in start:D
        for k in 1:K
            if k <= C
            gammas_DK[d, k] = (Y_DK[d,k] + alpha) / (e_phi_DK[d,k] + beta)
            else
                gammas_DK[d, k] = (Y_DK[d,k] + alpha) / max((e_phi_DK[d,k]+ beta - sum(e_phi_DK[d, 1:C] .* weights_KC[k, :].^d)), beta)
            end
        end
    end
    return(gammas_DK)
end
