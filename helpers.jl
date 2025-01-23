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



function update_theta_all(Theta_IK, Theta_IC, phi_VDK, lambdas_DK, phi_DK, Y_CV, w_KC, MIN_ORDER)
    V = size(Theta_IK, 1)
    @views for i = 1:V
        j = min(i+1, V)
        Theta_IK, phi_VDK, phi_DK, Theta_IC = update_theta_iC_D(
            i,
            Theta_IK,
            Theta_IC,
            phi_VDK,
            lambdas_DK,
            phi_DK,
            Y_CV[:, i],
            w_KC,
            j,
            MIN_ORDER,
            1e-80,
            1e-80,
            0,
        )
    end
    return(Theta_IK, phi_VDK, phi_DK, Theta_IC)
end

function test_stuff(Y_indices_test_D, Y_counts_test_D, lambdas_DK, Theta_IK, MIN_ORDER)
    log_pmf, auc =
        make_predictions(Y_indices_test_D, Y_counts_test_D, lambdas_DK, Theta_IK, MIN_ORDER)
    global llks = vcat(llks, log_pmf')
    times_np = convert(Array{Float64}, times)
    llks_np = convert(Array{Float64}, llks)
    npzwrite(
        "/net/projects/schein-lab/hood/Hypergraphs/results/" * directory * "/C$(C)K$(K)seed$(seed)RR.npz",
        Dict("times" => times_np, "llks" => llks_np),
    )
end

function evaluate_convergence(s, old_elbo, Y_counts_D, Y_indices_D, lambdas_DK, log_w_KC, Theta_IC, MIN_ORDER, CHECK_EVERY = 10)
    if s % CHECK_EVERY == 0
        println("Iteration: ", s)
        global likelihood =
            -compute_llk(Y_counts_D, Y_indices_D, lambdas_DK, log_w_KC, Theta_IC, MIN_ORDER)
        global change_elbo = likelihood - old_elbo
        global old_elbo = likelihood
    end
    global s += 1
end

function gradient(log_likelihood, θ, data)
    Zygote.gradient(t -> log_likelihood(t, data), θ)[1]
end



function reset_Y(D, V, C, K)
    Y_CV = zeros(C,V)
    Y_KC = zeros(K, C)
    Y_DK = zeros(D, K)
    return Y_CV, Y_KC, Y_DK
end

function init(K, C, D, V)
    w_KC, log_w_KC = init_W(K, C)
    lambdas_DK = ones(D, K)
    Theta_IC, Theta_IK = init_membership(V, C, K, D, w_KC)
    phi_DK, phi_VDK = compute_phi_DK(Theta_IK, D)
    phi_DK = log.(phi_DK)
    phi_VDK = log.(phi_VDK)
    return w_KC, log_w_KC, lambdas_DK, Theta_IC, Theta_IK, phi_VDK, phi_DK
end

function load_data(directory, MAX_ORDER = 25, MIN_ORDER = 2, test = false)
    data = npzread("data/" * directory * "/edges.npz")
    Y_indices_raw = data["edges"]
    Y_counts_raw = data["counts"]
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


function allocate_all(Y_indices_D, Y_counts_D, lambdas_DK, Theta_IK, w_KC, Theta_IC, MIN_ORDER, D)
    C = size(w_KC, 2)
    K = size(w_KC, 1)
    V = size(Theta_IK, 1)
    Y_CV, Y_KC, Y_DK = reset_Y(D, V, C, K)
    locker = Threads.SpinLock()
    @threads for d = MIN_ORDER:D
        Y_indices = Y_indices_D[d]
        Y_counts = Y_counts_D[d]
        lambdas_K = lambdas_DK[d, :]
        inds_V = inds_VD[d]
        Y_IK = allocate(Y_indices, Y_counts, lambdas_K, Theta_IK)
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
            alpha = ones(C)
            alpha[c] += 0
            w_KC[k, :] = rand(Dirichlet(alpha))
        end
    end
    return w_KC, log.(w_KC)
end

function impute(Y_indices, lambdas_K, Theta_IK)
    imputed_counts = zeros(size(Y_indices, 1), K) .+ log.(lambdas_K)'
    @views for d in axes(Y_indices, 2)
        imputed_counts .+= log.(Theta_IK[Y_indices[:, d], :])
    end
    imputed_counts = exp.(imputed_counts)
    @assert minimum(imputed_counts) >= 0
    return imputed_counts
end

function allocate(Y_indices, Y_counts, lambdas_K, Theta_IK)
    I = size(Theta_IK, 1)
    K = size(Theta_IK, 2)
    Y_IK = zeros(size(Y_indices, 1), K) .+ log.(lambdas_K)'
    for d in axes(Y_indices, 2)
        Y_IK += log.(Theta_IK[Y_indices[:, d], :])
    end
    Y_IK = exp.(Y_IK .- logsumexp(Y_IK, dims = 2)) .* (Y_counts)
    @assert sum(Y_IK) ≈ sum(Y_counts)
    return (Y_IK)
end

function gradient(Y_KC, Theta_IC, lambdas_DK, start, log_w_KC)
    Zygote.gradient(
        w -> compute_Welbo(Y_KC, w, Theta_IC, lambdas_DK, start),
        log_w_KC,
    )[1]
end

function hessian(Y_KC, Theta_IC, lambdas_DK, start, log_w_KC)
    g = log_w_KC -> gradient(Y_KC, Theta_IC, lambdas_DK, start, log_w_KC)  # Gradient as a function
    Zygote.jacobian(g, log_w_KC)  # Use ForwardDiff for second derivatives
end

function newton_raphson(Y_KC, Theta_IC, lambdas_DK, start, log_w_KC; tol=1e-6, max_iter=1)
    for i in 1:max_iter
        #println("Iteration $i: θ = $θ")
        g = gradient(Y_KC, Theta_IC, lambdas_DK, start, log_w_KC)      # Gradient
        H = hessian(Y_KC, Theta_IC, lambdas_DK, start, log_w_KC)       # Hessian
        Δθ = H \ g                                                     # Newton step (H^-1 * g)
        log_w_KC -= Δθ                                                 # Update parameters
        
        # Check for convergence
        if norm(Δθ) < tol || i == max_iter
            return log_w_KC
        end
    end
end

function optimize_Welbo(
    Y_KC,
    log_w_KC,
    Theta_IC,
    lambdas_DK,
    start,
    s,
    lr = 1e-5,
    steps = 1,
)
    for step = 1:steps
        grad_w_KC = Zygote.gradient(
            w -> compute_Welbo(Y_KC, w, Theta_IC, lambdas_DK, start),
            log_w_KC,
        )[1]
        #clip gradient
        grad_w_KC = min.(grad_w_KC, 1e5)
        grad_w_KC = max.(grad_w_KC, -1e5)
        grad_w_KC[isnan.(grad_w_KC)] .= 0
        log_w_KC .+= lr * grad_w_KC
    end
    return log_w_KC, log.(exp.(log_w_KC) .+ 1)
end


function optimize_WelboAdam(opt,
    Y_KC,
    log_w_KC,
    Theta_IC,
    lambdas_DK,
    start,
    s;
    steps = 1
)
    # Wrap parameters for Flux optimization
    log_w_KC_param = Flux.params(log_w_KC)
    for step = 1:steps
        loss() = compute_Welbo(Y_KC, log_w_KC, Theta_IC, lambdas_DK, start)
        Flux.Optimise.update!(opt, log_w_KC_param, Zygote.gradient(loss, log_w_KC_param))
        log_w_KC .= min.(log_w_KC, 1e5)
        log_w_KC .= max.(log_w_KC, -1e5)
        log_w_KC[isnan.(log_w_KC)] .= 0
    end

    # Return the updated log weights and their transformations
    return log_w_KC, log.(exp.(log_w_KC) .+ 1)
end


function compute_Welbo(Y_KC, log_w_KC, Theta_IC, lambdas_DK, start)
    D = size(lambdas_DK, 1)
    w_KC = log.(exp.(log_w_KC) .+ 1)
    Theta_IK = Theta_IC * w_KC'
    phi_DK = compute_phi_DK_min(Theta_IK, D)
    llk = 0
    llk -= sum(phi_DK[start:D, :] .* lambdas_DK[start:D, :])
    llk += sum(Y_KC .* log.(w_KC))
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


function compute_llk(Y_counts_D, Y_indices_D, lambdas_DK, log_w_KC, Theta_IC, start)
    w_KC = log.(exp.(log_w_KC) .+ 1)
    Theta_IK = Theta_IC * w_KC'
    llk = 0
    phi_DK = compute_phi_DK_min(Theta_IK, D)
    llk -= sum(phi_DK[start:D, :] .* lambdas_DK[start:D, :])
    for d = start:D
        Y_counts = Y_counts_D[d]
        Y_indices = Y_indices_D[d]
        rates_IK = zeros(size(Y_indices, 1), K) .+ log.(lambdas_DK[d, :])'
        for m = 1:size(Y_indices, 2)
            rates_IK = log.(Theta_IK[Y_indices[:, m], :]) + rates_IK
        end
        llk += sum(logsumexp(rates_IK, dims = 2) .* Y_counts)
    end
    println("Log-likelihood: $(llk)")
    return -llk
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
        col_sums[col_sums.==0] .= 1e-100
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
    lambdas_DK,
    phi_DK,
    Y_C,
    weights_KC,
    next_ind,
    start,
    alpha,
    beta,
    iter,
)
    C = size(Theta_IC, 2)
    K = size(Theta_IK, 2)
    old_theta = Theta_IK[i, :]
    D = size(phi_DK, 1)
    denominators = zeros(C)
    d_val = zeros(K)
    for d = start:D
        d_val .+= lambdas_DK[d, :] .* exp.(phi_VDK[i, d-1, :])
    end
    denominators .= transpose(weights_KC) * d_val
    new_theta_iC = (Y_C .+ alpha) ./ (denominators .+ beta) #shrinkage prior
    #new_theta_iC = new_theta_iC ./ sum(new_theta_iC)
    new_theta_iK = weights_KC * new_theta_iC
    phi_DK = update_phi_DK(old_theta, new_theta_iK, phi_DK, i, phi_VDK)
    Theta_IK[i, :] .= new_theta_iK
    Theta_IC[i, :] .= new_theta_iC
    phi_VDK = update_phi_DVK(Theta_IK[next_ind, :], phi_DK, phi_VDK, next_ind)
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
        test_indices = rand(1:n, n_test)
        train_indices = setdiff(1:n, test_indices)
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

function make_predictions(Y_indices_test_D, Y_counts_test_D, lambdas_DK, Theta_IK, start)
    D = length(Y_indices_D)
    predictions = zeros(0)
    log_pmf = zeros(0)
    prob_greater_0 = zeros(0)
    first_half = zeros(0)
    second_half = zeros(0)
    log_pmfs = zeros(D - start + 1)
    for d = start:D
        Y_indices = Y_indices_test_D[d]
        Y_counts = Y_counts_test_D[d]
        rates_IK = zeros(size(Y_indices, 1), K) .+ log.(lambdas_DK[d, :])'
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
        log_pmf = vcat(log_pmf, log_poisson)
        prob_greater_0 = vcat(prob_greater_0, prob_greater_0_d)
        first_half = vcat(first_half, first_half_d)
        second_half = vcat(second_half, second_half_d)
    end
    auc =
        (sum(first_half .> second_half) + 0.5 * sum(first_half .== second_half)) /
        length(first_half)
    return log_pmfs, auc
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

function update_phi_DVK(theta_k, phi_DK, phi_VDK, j)
    phi_VDK[j, 1, :] .= logsubexp.(phi_DK[1, :], log.(theta_k))
    @views for d = 2:D
        phi_VDK[j, d, :] .= logsubexp.(phi_DK[d, :], log.(theta_k) .+ phi_VDK[j, d-1, :])
    end
    return (phi_VDK)
end


function update_theta_iC_init(i, Theta_IK, Theta_IC, weights_KC)
    C = size(Theta_IC, 2)
    new_theta_iC = rand(Dirichlet(ones(C) .* 1000))
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

function update_lambda_DK(lambdas_DK, y_DK, phi_DK, alpha = 1e-2, beta = 1e-2)
    lambdas_DK = (y_DK .+ alpha) ./ (exp.(phi_DK) .+ beta)
    return (lambdas_DK)
end

function KLG(ta, tb, a, b)
    return (ta - 1) * digamma(ta) + log(tb) - ta - lgamma(ta) + lgamma(a) - a * log(b) -
           (a - 1) * digamma(ta) - (a - 1) * log(tb) + b / tb * ta#lgamma(a) - lgamma(ta) + ta*log(tb) - a*log(b) +  (ta - a)*(digamma(ta) - log(tb)) + (b - tb)*ta/tb
end
