function nextSample_RESTAPI(
    num_queries_to_allocate::Int64,
    design_trans_initial_phase::Vector{Float64},
    pep::Problem,
    opt_model::Model,
    cur_phase_idx::Int64,
    num_phases::Int64,
    cur_num_queries_in_cur_phase::Union{Nothing,Int64},
    cur_queryArm1Idx_allocation::Union{Nothing,Vector{Int64}},
    cur_queryArm2Idx_allocation::Union{Nothing,Vector{Int64}},
    cur_choiceOneZero_history::Union{Nothing,Vector{Bool}},
    cur_rt_history::Union{Nothing,Vector{Float64}},
    cur_active_arms::Union{Nothing,Vector{Int64}},
    phase_η::Int64,
    nondecision_time::Union{Nothing,Float64},
    subtract_nondecision_time::Bool,
    algorithm_name::String,
    design_method::String,
    debug::Bool)

    if subtract_nondecision_time
        @assert nondecision_time > 0
    end
    # d = length(pep.arms[1])
    K = length(pep.arms)
    # M = length(pep.queries)

    new_queryIdxs_allocation = nothing
    new_active_arms = nothing
    # Estimation, elimination, update design, rounding.
    if cur_phase_idx == 0 # initial round, before phase 1 data collection starts
        # println("\n\nInitial round => use cached design\n\n")
        distr = Categorical(design_trans_initial_phase)
        new_queryIdxs_allocation = rand(distr, num_queries_to_allocate)
        new_active_arms = collect(1:K)
    elseif cur_phase_idx <= num_phases
        # println("\n\nNot initial round => use computed design\n\n")
        dont_eliminate_arm = false
        if design_method in ["weakPref"] && cur_phase_idx == 1 # right after phase 1's data collection
            dont_eliminate_arm = true
        end
        (new_queryIdxs_allocation, new_active_arms) = compute_allocation(cur_num_queries_in_cur_phase, cur_queryArm1Idx_allocation, cur_queryArm2Idx_allocation, cur_choiceOneZero_history, cur_rt_history, cur_active_arms, num_queries_to_allocate, phase_η, pep, debug, nondecision_time, subtract_nondecision_time, algorithm_name, design_method, opt_model, dont_eliminate_arm)
    end
    if debug
        println("new_queryIdxs_allocation=", new_queryIdxs_allocation)
        pairs = [pep.queryIdx_2_armIdxPair[query_idx] for query_idx in new_queryIdxs_allocation]
        println("pairs=", pairs)
        println("new_active_arms=", new_active_arms)
    end

    new_queryArm1Idx_allocation = [pep.queryIdx_2_armIdxPair[query_idx][1] for query_idx in new_queryIdxs_allocation]
    new_queryArm2Idx_allocation = [pep.queryIdx_2_armIdxPair[query_idx][2] for query_idx in new_queryIdxs_allocation]
    new_queryArm1Idx_allocation = convert(Vector{Int64}, new_queryArm1Idx_allocation)
    new_queryArm2Idx_allocation = convert(Vector{Int64}, new_queryArm2Idx_allocation)

    if debug
        println("new_queryArm1Idx_allocation=", size(new_queryArm1Idx_allocation), "=", new_queryArm1Idx_allocation)
        println("new_queryArm2Idx_allocation=", size(new_queryArm2Idx_allocation), "=", new_queryArm2Idx_allocation)
    end

    return (new_queryArm1Idx_allocation, new_queryArm2Idx_allocation, new_active_arms)
end

function compute_allocation(
    cur_num_queries_in_cur_phase::Int64,
    cur_queryArm1Idx_allocation::Vector{Int64},
    cur_queryArm2Idx_allocation::Vector{Int64},
    cur_choiceOneZero_history_::Vector{Bool},
    cur_rt_history::Vector{Float64},
    cur_active_arms::Vector{Int64},
    num_queries_to_allocate::Int64,
    phase_η::Int64,
    pep::Problem,
    debug::Bool,
    nondecision_time::Union{Nothing,Float64},
    subtract_nondecision_time::Bool,
    algorithm_name::String,
    design_method::String,
    opt_model::Model,
    dont_eliminate_arm::Bool)

    @assert cur_num_queries_in_cur_phase > 0 # cannot be the initial round
    @assert cur_num_queries_in_cur_phase <= length(cur_queryArm1Idx_allocation)
    @assert length(cur_queryArm1Idx_allocation) == length(cur_queryArm2Idx_allocation)
    @assert iszero(cur_choiceOneZero_history_[cur_num_queries_in_cur_phase+1:end])
    @assert all(x -> x ∈ [0, 1], cur_choiceOneZero_history_)
    @assert iszero(cur_rt_history[cur_num_queries_in_cur_phase+1:end])

    K = length(pep.arms)
    d = length(pep.queries[1])
    M = length(pep.queries)

    # (1) Estimation
    θ_hat_cur_phase = nothing
    active_arm_pairs = Tuple{Int64,Int64}[]
    if debug
        for i1 = 1:length(cur_active_arms)
            for i2 = i1+1:length(cur_active_arms)
                k1 = cur_active_arms[i1]
                k2 = cur_active_arms[i2]
                push!(active_arm_pairs, (k1, k2))
            end
        end
    end

    arm_idx_pairs = [(cur_queryArm1Idx_allocation[i], cur_queryArm2Idx_allocation[i]) for i = 1:cur_num_queries_in_cur_phase]
    cur_queryIdx_history = [pep.armIdxPair_2_queryIdx[arm_idx_pair] for arm_idx_pair in arm_idx_pairs]
    cur_choiceOneZero_history = cur_choiceOneZero_history_[1:cur_num_queries_in_cur_phase]

    cur_choiceOneNegOne_history = cur_choiceOneZero_history .* 2 .- 1
    # cur_choiceOneTwo_history = 2 .- cur_choiceOneZero_history
    cur_rt_history = cur_rt_history[1:cur_num_queries_in_cur_phase]
    cur_query_history = zeros(Float64, cur_num_queries_in_cur_phase, d) # nxd
    for i = 1:cur_num_queries_in_cur_phase
        cur_query_history[i, :] = pep.queries[cur_queryIdx_history[i]]
    end

    if subtract_nondecision_time
        cur_rt_history = cur_rt_history .- nondecision_time
    end

    queryIdx_2_rts = Dict{Int64,Vector{Float64}}()
    queryIdx_2_choices = Dict{Int64,Vector{Int64}}()
    for (i, query_idx) in enumerate(cur_queryIdx_history)
        if !haskey(queryIdx_2_rts, query_idx)
            queryIdx_2_rts[query_idx] = []
            queryIdx_2_choices[query_idx] = []
        end
        push!(queryIdx_2_rts[query_idx], cur_rt_history[i])
        push!(queryIdx_2_choices[query_idx], cur_choiceOneNegOne_history[i])
    end
    if debug
        println("\n>>>>>>>>>>(1) Estimation>>>>>>>>>> ")
        println(algorithm_name)
        println("cur_num_queries_in_cur_phase=", cur_num_queries_in_cur_phase)
        println("cur_choiceOneZero_history=", cur_choiceOneZero_history)
        println("cur_choiceOneNegOne_history=", cur_choiceOneNegOne_history)
        println("cur_rt_history=", round.(cur_rt_history; digits=3))
        for (query_idx, rts) in queryIdx_2_rts
            arm_idx_pair = pep.queryIdx_2_armIdxPair[query_idx]
            println("Query=", query_idx, "=(", arm_idx_pair[1], ", ", arm_idx_pair[2], ") is sampled ", length(rts), " times.")
        end
    end

    if algorithm_name ∈ ["GLM"]
        θ_hat_cur_phase = solve_GLM_in_Algorithms(cur_query_history, cur_choiceOneZero_history, cur_choiceOneNegOne_history, cur_queryIdx_history, opt_model, pep.queries)
    elseif algorithm_name ∈ ["LM", "Chiong24Lemma1", "Wagenmakers07Eq5"]
        X = nothing
        Xy = zeros(Float64, d)
        if algorithm_name ∈ ["LM"]
            X = zeros(Float64, length(cur_queryIdx_history), d)
            for (i, query_idx) in enumerate(cur_queryIdx_history)
                EChoice = mean(queryIdx_2_choices[query_idx])
                ERt = mean(queryIdx_2_rts[query_idx])

                X[i, :] = pep.queries[query_idx]
                Xy_incre = pep.queries[query_idx] .* EChoice / ERt
                Xy += Xy_incre
            end
        elseif algorithm_name ∈ ["Wagenmakers07Eq5"]
            X = zeros(Float64, length(cur_queryIdx_history), d)
            for (i, query_idx) in enumerate(cur_queryIdx_history)
                EChoice = mean(queryIdx_2_choices[query_idx])
                ERt = mean(queryIdx_2_rts[query_idx])
                EChoice_zeroOne = (EChoice + 1) / 2
                @assert EChoice_zeroOne >= 0 && EChoice_zeroOne <= 1

                # Eq.5 in Wagenmakers, E. J., Van Der Maas, H. L., & Grasman, R. P. (2007). An EZ-diffusion model for response time and accuracy. Psychonomic bulletin & review, 14(1), 3-22.
                # The following trick is from under Fig.6 in Wagenmakers, E. J., Van Der Maas, H. L., & Grasman, R. P. (2007). An EZ-diffusion model for response time and accuracy. Psychonomic bulletin & review, 14(1), 3-22.
                if EChoice_zeroOne == 1
                    EChoice_zeroOne = 1 - 1 / (2 * (length(queryIdx_2_choices[query_idx])))
                elseif EChoice_zeroOne == 0
                    EChoice_zeroOne = 1 / (2 * (length(queryIdx_2_choices[query_idx])))
                end
                logit = log(EChoice_zeroOne / (1 - EChoice_zeroOne))

                X[i, :] = pep.queries[query_idx]
                Xy_incre = pep.queries[query_idx] .* logit
                Xy += Xy_incre
            end
        elseif algorithm_name ∈ ["Chiong24Lemma1"]
            X = zeros(Float64, length(cur_queryIdx_history), d)
            for (i, query_idx) in enumerate(cur_queryIdx_history)
                EChoice = mean(queryIdx_2_choices[query_idx])
                ERt = mean(queryIdx_2_rts[query_idx])
                EChoice_zeroOne = (EChoice + 1) / 2
                @assert EChoice_zeroOne >= 0 && EChoice_zeroOne <= 1

                if EChoice_zeroOne == 1
                    EChoice_zeroOne = 1 - 1 / (2 * (length(queryIdx_2_choices[query_idx])))
                elseif EChoice_zeroOne == 0
                    EChoice_zeroOne = 1 / (2 * (length(queryIdx_2_choices[query_idx])))
                end

                # Lemma 1 in Xiang Chiong, K., Shum, M., Webb, R., & Chen, R. (2024). Combining choice and response time data: A drift-diffusion model of mobile advertisements. Management Science, 70(2), 1238-1257.
                logit = log(EChoice_zeroOne / (1 - EChoice_zeroOne))
                incre = sqrt(EChoice / (2 * ERt) * logit)
                if EChoice_zeroOne < 0.5
                    incre = -1 * incre
                end

                X[i, :] = pep.queries[query_idx]
                Xy_incre = pep.queries[query_idx] .* incre
                Xy += Xy_incre
            end
        end
        try
            θ_hat_cur_phase = pinv(X' * X) * Xy
        catch err # Follow GLM's solution
            @error "ERROR: solve LM via least squares failed: " err
            θ_hat_cur_phase = qr(X' * X, Val(true)) \ Xy
        end
    else
        error("Invalid algorithm_name=" * algorithm_name)
    end
    @assert !isnothing(θ_hat_cur_phase)
    if debug
        println("θ_hat_cur_phase=", round.(θ_hat_cur_phase; digits=3))
        println("<<<<<<<<<<<<<<<<<<<<")
    end

    # (2) Elimination
    reward_armIdx = [(pep.arms[arm_idx]' * θ_hat_cur_phase, arm_idx) for arm_idx in cur_active_arms]
    reward_armIdx = shuffle(reward_armIdx) # random tie breaking for sorting
    reward_armIdx_sorted = sort(reward_armIdx, rev=true)

    new_active_arms = Int64[]
    if dont_eliminate_arm
        # Burn-in phase with estimation without elimination
        new_active_arms = collect(1:K)
        if debug
            println("weakPref's burn in")
        end
    else
        # [1]'s Alg.1: ceil(|At|/η)
        middle_idx = ceil(Int64, length(cur_active_arms) / phase_η)
        @assert middle_idx >= 1
        new_active_arms = [x[2] for x in (reward_armIdx_sorted[1:middle_idx])]
    end
    if debug
        println(">>>>>>>>>>(2) Elimination>>>>>>>>>> ", algorithm_name)
        println("cur_active_arms=", cur_active_arms)
        println("reward_armIdx_sorted=")
        for (reward, idx) in reward_armIdx_sorted
            @printf "(%.3f, %d) " reward idx
        end
        println("\n=> new_active_arms=", new_active_arms, ", len=", length(new_active_arms))
        println("sorted new_active_arms=", sort(new_active_arms))
        println("<<<<<<<<<<<<<<<<<<<<")
    end

    if length(new_active_arms) == 1 # termination
        return ([], new_active_arms)
    end

    # (3) Update design
    query_weights = ones(Float64, M)
    if design_method ∈ ["weakPref"]
        # Alg.1 in Jun, K. S., Jain, L., Mason, B., & Nassif, H. (2021, July). Improved confidence bounds for the linear logistic model and applications to bandits. In International Conference on Machine Learning (pp. 5148-5157). PMLR.
        for (query_idx, query) in enumerate(pep.queries)
            # https://heliosphan.org/logistic-fn-gradient.html
            @assert !isnothing(θ_hat_cur_phase)
            reward_ = query' * θ_hat_cur_phase
            μdot = exp(-reward_) / (1 + exp(-reward_))^2
            if abs(reward_) > log(1e15)
                μdot = 0
            end
            if isnan(μdot) || isinf(μdot)
                @error "μdot=" μdot
                println("reward_=", reward_)
                println("e(-reward_)=", exp(-reward_))
                println("1+e(-reward_)=", (1 + exp(-reward_)))
                error(algorithm_name, ": μdot=nan/inf")
            end
            query_weights[query_idx] = μdot
        end
    end

    # min_λ max_{y\in Zt-Zt} \|y\|^2_{(Σ_{x\in X} λx x x^T)^-1}.
    # Y=|Zt|*(|Zt|-1) x d, Y_arm_idx_pairs=|Zt|*(|Zt|-1) x 2
    Y, Y_arm_idx_pairs = build_Y(new_active_arms, pep.arms)
    query_idxs = 1:M
    queries = zeros(Float64, length(query_idxs), d)
    for (i, query_idx) in enumerate(query_idxs)
        queries[i, :] = pep.queries[query_idx]
    end
    @assert size(Y, 2) == d == size(queries, 2)
    @assert size(queries, 1) == length(query_idxs)
    @assert length(query_idxs) >= 1
    @assert size(Y, 1) >= 1

    # I. Original hyper-parameters
    # https://github.com/AndreaTirinzoni/bandit-elimination/blob/39aa1b9fe36759496119367178a097240b3c5571/sampling_rules.jl#L509
    # =App.F [2]
    # max_iter = 5e3
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
    design_, rho_ = optimal_allocation(queries, Y, query_costs_, query_weights_, max_iter, epsilon, weight_cutoff)
    # Convert the design_ back to full M numbers.
    cur_design = zeros(Float64, M)
    for (i, query_idx) in enumerate(query_idxs)
        cur_design[query_idx] = design_[i]
    end

    if debug
        println(">>>>>>>>>>(3) Design>>>>>>>>>> ", algorithm_name)
        # println("cur_design=", round.(cur_design; digits=3))
        # println("cur_design=", cur_design)
        tmp = 0
        for m = 1:M
            if cur_design[m] > 1e-5
                println("m=", m, ": ", cur_design[m])
                tmp += cur_design[m]
            end
        end
        println("Sum of >1e-5 weights=", tmp)
        println("<<<<<<<<<<<<<<<<<<<<")
    end

    # (4) Rounding
    distr = Categorical(cur_design)
    new_queryIdxs_allocation = rand(distr, num_queries_to_allocate)

    if debug
        pairs = [pep.queryIdx_2_armIdxPair[query_idx] for query_idx in new_queryIdxs_allocation]
        println(">>>>>>>>>>(4) Rounding>>>>>>>>>> ", algorithm_name)
        println("new pairs to sample=", size(pairs), "=", pairs)
        println("<<<<<<<<<<<<<<<<<<<<")
    end

    return (new_queryIdxs_allocation, new_active_arms)
end