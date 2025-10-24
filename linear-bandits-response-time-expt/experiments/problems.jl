function produce_queries(arms, duel_trans)
    for arm in arms
        @assert count(x -> (abs(x) > 1e-5), arm) > 0
    end
    armIdxPair_2_queryIdx = Dict{Tuple{Int64,Int64},Int64}()
    queryIdx_2_armIdxPair = Array{Tuple{Int64,Int64}}(undef, 0)
    if !duel_trans
        queries = arms
        query_idx = 1
        for k in eachindex(arms)
            armIdxPair_2_queryIdx[(k, 0)] = query_idx
            push!(queryIdx_2_armIdxPair, (k, 0))
            query_idx += 1
        end
    else
        query_idx = 1
        queries = Vector{Float64}[] # transductive
        for k1 in eachindex(arms)
            for k2 = k1+1:length(arms)
                x = arms[k1] - arms[k2]
                push!(queries, x)
                armIdxPair_2_queryIdx[(k1, k2)] = query_idx
                armIdxPair_2_queryIdx[(k2, k1)] = query_idx
                push!(queryIdx_2_armIdxPair, (k1, k2))
                query_idx += 1
            end
        end
    end
    for query in queries
        @assert count(x -> (abs(x) > 1e-5), query) > 0
    end
    for (k, v) in armIdxPair_2_queryIdx
        # println("Arms ", k, "=query ", v)
        @assert (queryIdx_2_armIdxPair[v] == k) || (queryIdx_2_armIdxPair[v] == (k[2], k[1]))
    end
    for (k, v) in enumerate(queryIdx_2_armIdxPair)
        # println("Query ", k, "=arms ", v)
        @assert armIdxPair_2_queryIdx[v] == k
    end
    return queries, armIdxPair_2_queryIdx, queryIdx_2_armIdxPair
end

function problem4_sphere(duel_trans::Bool, d::Int64, K::Int64, init_seed::Int64, scale_z::Union{Float64,Int64})
    Random.seed!(init_seed)
    # /Users/shenli/Dropbox (MIT)/1111111111/616-bandit/--linear/-fixedBudget structLinear 23 Li - Optimal Exploration is no harder than Thompson Sampling.pdf
    # d = 6
    # K = 20
    α = 0.01

    arms = Vector{Float64}[]
    for k = 1:K
        v = rand(d)
        v = v / norm(v, 2)
        push!(arms, v)
    end

    val = -Inf
    val_pair = nothing
    for k1 = 1:K
        for k2 = k1+1:K
            # /Users/shenli/Dropbox (MIT)/MIT/projects/sample complexity-MIT-220704/231016 - dueling bandit - latex/v10 fixed-budget - empirical.pdf
            tmp = arms[k1]' * arms[k2]
            if tmp > val
                val_pair = (k1, k2)
                val = tmp
            end
        end
    end
    θ = arms[val_pair[1]] + α * (arms[val_pair[2]] - arms[val_pair[1]])
    @assert argmax([arms[k]' * θ for k = 1:K]) == val_pair[1]

    # https://github.com/shenlirobot/dueling_bandit_reaction_time/commit/15c71d471e067d6667272fecc0f85055b498909b
    # θ .*= 100

    for (i, arm) in enumerate(arms)
        arms[i] = arm .* scale_z
    end

    @assert count(x -> (abs(x) > 1e-5), θ) > 0
    @assert length(θ) == d
    queries, armIdxPair_2_queryIdx, queryIdx_2_armIdxPair = produce_queries(arms, duel_trans)
    @assert isinteger(scale_z * 10) # ensure scale_z has only 1 decimal
    problem_name = "4d" * string(d) * "K" * string(K) * "Sz" * @sprintf("%.1f", scale_z)
    return θ, arms, queries, armIdxPair_2_queryIdx, queryIdx_2_armIdxPair, problem_name
end

function problem7_Clithero_v3(subjectOfInterest_idx::Int64, data_path::String, init_seed::Int64, duel_trans::Bool)
    @assert !isnothing(init_seed)
    Random.seed!(init_seed + subjectOfInterest_idx)

    # https://github.com/JuliaIO/JLD2.jl
    subjectIdx_2_params = load(data_path, "subjectIdx_2_params")
    num_subjects = length(subjectIdx_2_params) # 31
    num_items = length(subjectIdx_2_params[subjectOfInterest_idx]["νs"]) # 17
    @assert 1 <= subjectOfInterest_idx <= num_subjects

    arms_are_standard_bandits = true
    d = num_items
    θ = subjectIdx_2_params[subjectOfInterest_idx]["νs"]
    arms = Vector{Float64}[]

    for item_idx = 1:num_items
        z = zeros(num_items)
        z[item_idx] = 1
        push!(arms, z)

        reward = transpose(z) * θ
        @assert abs(reward - subjectIdx_2_params[subjectOfInterest_idx]["νs"][item_idx]) < 1e-10
    end
    if arms_are_standard_bandits
        for arm in arms
            @assert count(x -> (x == 0), arm) == (d - 1)
            @assert count(x -> (x == 1), arm) == 1
        end
    end
    @assert length(θ) == d
    @assert length(arms) == num_items
    @assert all(x -> (length(x) == d), arms)

    queries, armIdxPair_2_queryIdx, queryIdx_2_armIdxPair = produce_queries(arms, duel_trans)
    problem_name = "7s" * string(subjectOfInterest_idx)
    return θ, arms, queries, armIdxPair_2_queryIdx, queryIdx_2_armIdxPair, problem_name
end

function problem8_Krajbich(subjectOfInterest_idx::Int64, data_path::String, init_seed::Int64)
    @assert !isnothing(init_seed)
    Random.seed!(init_seed + subjectOfInterest_idx)

    # https://github.com/JuliaIO/JLD2.jl
    subjectIdx_2_params = load(data_path, "subjectIdx_2_params")
    num_subjects = length(subjectIdx_2_params) # 39
    @assert 1 <= subjectOfInterest_idx <= num_subjects

    θ = collect(-10:10)
    θ = θ .* subjectIdx_2_params[subjectOfInterest_idx]["l"]
    θ = shuffle(θ)
    d = length(θ)

    arms = Vector{Float64}[]
    for i = 1:d
        z = zeros(d)
        z[i] = 1
        push!(arms, z)
    end

    @assert length(θ) == length(arms[1])
    duel_trans = true
    queries, armIdxPair_2_queryIdx, queryIdx_2_armIdxPair = produce_queries(arms, duel_trans)
    problem_name = "8s" * string(subjectOfInterest_idx)
    return θ, arms, queries, armIdxPair_2_queryIdx, queryIdx_2_armIdxPair, problem_name
end

function problem12_foodrisk(subjectOfInterest_idx::Int64, data_path::String)
    # https://github.com/JuliaIO/JLD2.jl
    subjectIdx_2_params = load(data_path, "subjectIdx_2_params")
    num_subjects = length(subjectIdx_2_params) # 42
    @assert 1 <= subjectOfInterest_idx <= num_subjects

    θ = subjectIdx_2_params[subjectOfInterest_idx]["θ"]
    arms = convert(Vector{Vector{Float64}}, subjectIdx_2_params[subjectOfInterest_idx]["zs"])

    arm_idxs_duplicate = Int64[]
    for (a1, arm1) in enumerate(arms)
        for a2 = a1+1:length(arms)
            if norm(arm1 - arms[a2], Inf) < 1e-5
                push!(arm_idxs_duplicate, a2)
            end
        end
    end
    println("Before removing duplicates, K=", length(arms))
    println("arm_idxs_duplicate=", arm_idxs_duplicate)
    arms = [a for (i, a) in enumerate(arms) if i ∉ arm_idxs_duplicate]
    println("After removing duplicates, K=", length(arms))

    # arms = shuffle(arms)

    duel_trans = true
    queries, armIdxPair_2_queryIdx, queryIdx_2_armIdxPair = produce_queries(arms, duel_trans)
    problem_name = "12s" * string(subjectOfInterest_idx)
    return θ, arms, queries, armIdxPair_2_queryIdx, queryIdx_2_armIdxPair, problem_name
end