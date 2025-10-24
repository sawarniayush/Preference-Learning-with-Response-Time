function experimental_design(problem_config::Tuple{Vector{Float64},Vector{Vector{Float64}},Vector{Vector{Float64}},Dict{Tuple{Int64,Int64},Int64},Vector{Tuple{Int64,Int64}},String}, design_method::String, DDM_barrier_from_0::Union{Nothing,Int64,Float64}=nothing)
    @assert design_method ∈ ["trans", "weakPref"]

    (θ, arms, queries, armIdxPair_2_queryIdx, queryIdx_2_armIdxPair, problem_name) = problem_config
    d = length(arms[1])
    K = length(arms)
    M = length(queries)

    query_weights = ones(Float64, M)
    if design_method == "weakPref"
        # Alg.1 in Jun, K. S., Jain, L., Mason, B., & Nassif, H. (2021, July). Improved confidence bounds for the linear logistic model and applications to bandits. In International Conference on Machine Learning (pp. 5148-5157). PMLR.
        for (query_idx, query) in enumerate(queries)
            # https://heliosphan.org/logistic-fn-gradient.html
            reward = 2 * DDM_barrier_from_0 * query' * θ

            # https://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
            if reward > 0
                μdot = exp(-reward) / (1 + exp(-reward))^2
            else
                μdot = exp(reward) / (1 + exp(reward))^2
            end
            if isnan(μdot)
                println("μdot=", μdot, ", is nan.\nquery=", round.(query; digits=3), ", θ=", round.(θ; digits=3), ", reward=", reward)
                error("NaN")
            end
            if isinf(μdot)
                println("μdot=", μdot, ", is inf.\nquery=", round.(query; digits=3), ", θ=", round.(θ; digits=3))
                error("Inf")
            end
            query_weights[query_idx] = μdot
        end
        query_weights = query_weights ./ norm(query_weights, 2)
    end

    # min_λ max_{y\in Y} \|y\|^2_{(Σ_{x\in queries} λx x x^T)^-1}.
    # Y=Zt-Zt, queries=X.
    # Y=|Zt|*(|Zt|-1) x d, Y_arm_idx_pairs=|Zt|*(|Zt|-1) x 2
    Y, _ = build_Y(collect(1:K), arms)
    @assert size(Y, 2) == d
    @assert size(Y, 1) >= 1
    query_idxs = 1:M

    # I. Original hyper-parameters
    # https://github.com/AndreaTirinzoni/bandit-elimination/blob/39aa1b9fe36759496119367178a097240b3c5571/sampling_rules.jl#L509
    # max_iter = Int(5e3)
    # epsilon = 1e-2
    # weight_cutoff = 1e-5
    # II. Finer hyper-parameters
    # max_iter = Int(1e5)
    # epsilon = 1e-3
    # weight_cutoff = 1e-5
    # III. Coarser hyper-parameters
    # II is too slow. So maybe we can relax the weight_cutoff, which is 5x faster than II.
    max_iter = Int(1e5)
    epsilon = 1e-2
    weight_cutoff = 1e-5

    query_costs = ones(Float64, M)
    query_costs_ = [query_costs[i] for i in query_idxs]
    query_weights_ = [query_weights[i] for i in query_idxs]
    queries_mat = zeros(Float64, length(query_idxs), d)
    for (i, query_idx) in enumerate(query_idxs)
        queries_mat[i, :] = queries[query_idx]
    end
    design_, rho_ = optimal_allocation(queries_mat, Y, query_costs_, query_weights_, max_iter, epsilon, weight_cutoff)
    # Convert the design_ back to full M numbers.
    design = zeros(Float64, M)
    for (i, query_idx) in enumerate(query_idxs)
        design[query_idx] = design_[i]
    end
    design_over_cost = design ./ query_costs
    design_over_cost_normalized = design_over_cost ./ sum(design_over_cost)
    return design_over_cost_normalized
end

function build_Y(active_arm_idxs::Vector{Int64}, arms::Vector{Vector{Float64}})
    # Y = active_arms - active_arms
    d = length(arms[1])
    n = length(active_arm_idxs)
    Y = zeros(Float64, Int64(n * (n - 1) / 2), d)
    Y_arm_idx_pairs = Vector{Tuple{Int64,Int64}}(undef, Int64(n * (n - 1) / 2))
    counter = 1
    for i = 1:n
        for j = i+1:n
            arm1_idx = active_arm_idxs[i]
            arm2_idx = active_arm_idxs[j]
            Y[counter, :] = arms[arm1_idx] - arms[arm2_idx]
            Y_arm_idx_pairs[counter] = (arm1_idx, arm2_idx)
            counter += 1
        end
    end
    return Y, Y_arm_idx_pairs
end

# https://github.com/AndreaTirinzoni/bandit-elimination/blob/39aa1b9fe36759496119367178a097240b3c5571/sampling_rules.jl#L509
# min_λ max_{z,z'\in Y} \|z-z'\|^2_{(Σ_{x\in queries} λx x x^T)^-1}.
function optimal_allocation(queries::Union{Matrix{Float64},Transpose{Float64,Matrix{Float64}}}, Y::Matrix{Float64}, query_costs::Vector{Float64}, query_weight::Vector{Float64}, max_iter::Int64=Int(5e3), epsilon::Float64=1e-2, weight_cutoff::Float64=1e-5)
    M, d = size(queries)
    @assert size(Y, 2) == d # Y=|Y|xd
    design = ones(M) ./ M
    rho = 0

    converged = false
    m_2_outerProduct = zeros(Float64, M, d, d)
    for m = 1:M
        m_2_outerProduct[m, :, :] = queries[m, :] * transpose(queries[m, :])
    end
    for count in 1:max_iter
        A = sum([(design[m] / query_costs[m] * query_weight[m]) .* m_2_outerProduct[m, :, :] for m = 1:M])

        Ainvhalf = nothing
        try
            A_inv = pinv(A)
            @assert !any(x -> isnan(x), A_inv)
            U, D, V = svd(A_inv)
            Ainvhalf = U * Diagonal(sqrt.(D)) * transpose(V)
            # println(maximum(abs.(pinv(A) - Ainvhalf * Ainvhalf)))
        catch err
            @warn "ERROR: optimal_allocation's pinv failed: " err
            # https://yanagimotor.github.io/posts/2021/06/blog-post-lapack/
            U2 = nothing
            D2 = nothing
            V2 = nothing
            try
                U2, D2, V2 = svd(A, alg=LinearAlgebra.DivideAndConquer())
            catch e2
                U2, D2, V2 = svd(A, alg=LinearAlgebra.QRIteration())
            end
            @assert maximum(abs.(A - U2 * Diagonal(D2) * transpose(V2))) < 1e-8
            # https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.pinv
            zero_idxs = [i for i = 1:d if D2[i] < min(size(A)...) * eps()]
            D2_ = 1 ./ D2
            D2_[zero_idxs] .= 0
            Ainvhalf2 = V2 * Diagonal(sqrt.(D2_)) * transpose(U2)
            # println(maximum(abs.(pinv(A) - Ainvhalf2 * Ainvhalf2)))
            Ainvhalf = Ainvhalf2
        end

        # Y=\Y(\Z\hat_t)=(|cur_active_arms|^2 x d) in [1]'s Alg.1
        newY = (Y * Ainvhalf) .^ 2 # |Y|xd
        rho = sum(newY, dims=2)[:, 1]

        idx = argmax(rho)
        y = Y[idx, :]
        g = 1 ./ query_costs .* query_weight .* vec((queries * Ainvhalf * Ainvhalf * y) .^ 2)
        g_idx = argmax(g)

        gamma = 2 / (count + 2)
        design_update = copy(design)
        design_update .*= -gamma
        design_update[g_idx] += gamma

        relative = norm(design_update) / (norm(design))

        design .+= design_update

        if relative < epsilon
            converged = true
            break
        end
    end
    if !converged
        @warn "[optimal_allocation]: not converged."
    end
    # https://github.com/AndreaTirinzoni/bandit-elimination/blob/39aa1b9fe36759496119367178a097240b3c5571/sampling_rules.jl#L544
    design[design.<weight_cutoff] .= 0
    design /= sum(design)
    return design, maximum(rho)
end

function solve_GLM_in_Algorithms(cur_query_history::Matrix{Float64}, cur_choiceOneZero_history::Vector{Bool}, cur_choiceOneNegOne_history::Vector{Int64}, cur_queryIdx_history::Vector{Int64}, opt_model::Union{Nothing,Model}, queries::Vector{Vector{Float64}})
    d = length(queries[1])
    M = length(queries)

    # Eq.5 in Clithero, J. A. (2018). Improving out-of-sample predictions using response times and a model of the decision process. Journal of Economic Behavior & Organization, 148, 344-375.
    θ_hat_cur_phase = nothing
    scale_param = 1
    model_name = "logit"
    try
        verbose = false
        θ_hat_cur_phase = solve_GLM_via_GLM_package(model_name, cur_query_history, cur_choiceOneZero_history, scale_param, verbose)
    catch err
        # @warn "ERROR: solve_GLM_via_GLM_package failed: " err
        try
            if isnothing(opt_model)
                opt_model = Model(Ipopt.Optimizer)
                set_silent(opt_model)
            end
            θ_hat_cur_phase = solve_GLM_via_optimizatiaon(model_name, opt_model, cur_query_history, cur_choiceOneNegOne_history, scale_param)
        catch err
            # https://docs.julialang.org/en/v1/base/base/#Core.MethodError
            # https://scls.gitbooks.io/ljthw/content/_chapters/11-ex8.html
            if isa(err, SystemError) || isa(err, LAPACKException)
                # https://discourse.julialang.org/t/how-to-find-the-error-line-using-try-catch-statements/65511/2
                @error "ERROR: solve_GLM_via_optimizatiaon failed: " err

                # Use OLS instead
                gram_matrix = nothing
                gram_matrix_inv = nothing
                gram_matrix = zeros(Float64, d, d)
                tmp = counter(cur_queryIdx_history)
                for m = 1:M
                    @assert tmp[m] >= 0 && Int(tmp[m]) == tmp[m]
                    gram_matrix += tmp[m] * queries[m] * queries[m]'
                end
                cur_rx = zeros(d)
                for i in eachindex(cur_choiceOneNegOne_history)
                    cur_rx += cur_query_history[i, :] .* cur_choiceOneNegOne_history[i]
                end
                try
                    gram_matrix_inv = pinv(gram_matrix)
                    θ_hat_cur_phase = gram_matrix_inv * cur_rx
                catch eee
                    @error "ERROR: solve GLM via least squares failed: " err
                    # To prevent <From worker 2: err = LAPACKException(1)>:
                    # I. https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.pinv
                    # gram_matrix_inv = pinv(gram_matrix, rtol=sqrt(eps(real(float(oneunit(eltype(gram_matrix)))))))
                    # II.
                    # gram_matrix_inv = pinv(gram_matrix + I * 1e-5)
                    # θ_hat_cur_phase = gram_matrix_inv * cur_rx
                    # III. https://discourse.julialang.org/t/pseudo-inverse-and-the-backslash-operator/27882
                    θ_hat_cur_phase = qr(gram_matrix, Val(true)) \ cur_rx
                end
            else
                rethrow()
            end
        end
    end
    return θ_hat_cur_phase
end