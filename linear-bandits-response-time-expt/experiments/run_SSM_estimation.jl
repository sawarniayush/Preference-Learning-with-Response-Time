using Distributed
num_cores = 1
num_cores = 12
if num_cores > 1
    addprocs(num_cores - 1)
end
@everywhere using JLD2, Printf, IterTools, Random, DataStructures
@everywhere using LinearAlgebra, Distributions, GLM
@everywhere using Plots, LaTeXStrings, Measures, Base
@everywhere using JuMP, Ipopt, YAML
@everywhere include("../src/problems.jl")
@everywhere include("../experiments/problems.jl")
@everywhere include("../src/algorithms_utils.jl")
@everywhere include("../src/GLM.jl")
@everywhere include("../src/DDM_local.jl")


function main()
    seed_problem_definition = 123
    seed_dataset = 123
    base_path = @__DIR__
    base_path = base_path * "/"

    # ---------------------------------------------
    methods = [
        ("GLM", "trans"), # Choices only + transductive design
        ("GLM", "weakPref"), # Choices only + weak preference design (Jun, K. S., Jain, L., Mason, B., & Nassif, H. (2021, July). Improved confidence bounds for the linear logistic model and applications to bandits. In International Conference on Machine Learning (pp. 5148-5157). PMLR.)
        ("LM", "trans"), # Our proposed method + transductive design
    ]
    use_SSM_simulation = true # slow, but works for all range of α
    problem_definition_d = 5
    problem_definition_K = 10
    budgets = [50]

    repeats_problemDefinition = 100
    repeats_dataset = 100

    DDM_barrier_from_0s = collect(0.1:0.1:2.5)
    scale_zs = collect(0.1:0.1:0.9)
    scale_zs = vcat(scale_zs, collect(1:5:101))
    scale_zs = vcat(scale_zs, collect(200:100:1000))

    # repeats_problemDefinition = 5 # for debug
    # repeats_dataset = 5 # for debug
    # DDM_barrier_from_0s = [0.5, 1.5, 2.5] # for debug
    # scale_zs = [0.5, 1, 10] # for debug
    # ---------------------------------------------

    println("DDM_barrier_from_0s=", collect(DDM_barrier_from_0s))
    println("scale_zs=", collect(scale_zs))
    println("Total combinations: ", length(DDM_barrier_from_0s) * length(scale_zs))

    if num_cores == 1
        datas = map(
            scale_z -> prepare_one_scale_z(scale_z, DDM_barrier_from_0s, repeats_problemDefinition, problem_definition_d, problem_definition_K, seed_problem_definition),
            scale_zs
        )
    else
        datas = pmap(
            scale_z -> prepare_one_scale_z(scale_z, DDM_barrier_from_0s, repeats_problemDefinition, problem_definition_d, problem_definition_K, seed_problem_definition),
            scale_zs
        )
    end
    @assert size(datas) == (length(scale_zs),)
    scaleZ_2_problemConfigs = Dict()
    scaleZ_DDMΒarrierFrom0_2_designss = Dict()
    for (scale_z_idx, scale_z) in enumerate(scale_zs)
        scale_z_, problem_configs, DDMΒarrierFrom0_2_designss = datas[scale_z_idx]
        @assert scale_z == scale_z_
        scaleZ_2_problemConfigs[scale_z] = problem_configs
        for (b, v) in DDMΒarrierFrom0_2_designss
            scaleZ_DDMΒarrierFrom0_2_designss[(scale_z, b)] = v
        end
    end

    for scale_z in scale_zs
        for DDM_barrier_from_0 in DDM_barrier_from_0s
            @assert length(scaleZ_2_problemConfigs[scale_z]) == repeats_problemDefinition
            @assert length(scaleZ_DDMΒarrierFrom0_2_designss[(scale_z, DDM_barrier_from_0)]) == repeats_problemDefinition
        end
    end

    DDMBarrierFrom0_scalez_budget_method_2_error = Dict()
    for scale_z in scale_zs
        for DDM_barrier_from_0 in DDM_barrier_from_0s
            problem_configs = scaleZ_2_problemConfigs[scale_z]
            designss = scaleZ_DDMΒarrierFrom0_2_designss[(scale_z, DDM_barrier_from_0)]

            if DDM_barrier_from_0 < 1
                use_SSM_simulation_integration_Δt = 1e-5
            else
                use_SSM_simulation_integration_Δt = 1e-4
            end
            @time begin
                for budget in budgets
                    method_2_errorName_2_error = run_one_trial(Float64(DDM_barrier_from_0), scale_z, budget, use_SSM_simulation, use_SSM_simulation_integration_Δt, methods, seed_dataset, repeats_problemDefinition, repeats_dataset, problem_configs, designss, base_path)

                    if !isnothing(method_2_errorName_2_error)
                        for (method, errorName_2_error) in method_2_errorName_2_error
                            k = (DDM_barrier_from_0, scale_z, budget, method)
                            DDMBarrierFrom0_scalez_budget_method_2_error[k] = errorName_2_error
                        end
                    end
                end
            end
        end
    end

    result_path = base_path * "run_SSM_estimation/"
    if !isdir(result_path)
        mkpath(result_path)
    else
        # Remove all files inside the folder
        for file in readdir(result_path)
            file_path = joinpath(result_path, file)
            if isfile(file_path)  # Ensure it's a file before deleting
                rm(file_path)
            end
        end
    end

    path = result_path * "run_SSM_estimation.dat"
    @save path DDMBarrierFrom0_scalez_budget_method_2_error DDM_barrier_from_0s scale_zs budgets methods

    tmp = Dict()
    tmp["DDMBarrierFrom0_scalez_budget_method_2_error"] = DDMBarrierFrom0_scalez_budget_method_2_error
    tmp["DDM_barrier_from_0s"] = DDM_barrier_from_0s
    tmp["scale_zs"] = scale_zs
    tmp["budgets"] = budgets
    tmp["methods"] = methods
    path = result_path * "run_SSM_estimation.yaml"
    YAML.write_file(path, tmp)
end

@everywhere function prepare_one_scale_z(scale_z, DDM_barrier_from_0s, repeats_problemDefinition, problem_definition_d, problem_definition_K, seed_problem_definition)
    problem_configs = []
    DDMΒarrierFrom0_2_designss = Dict()
    for problem_definition_idx = 1:repeats_problemDefinition
        θ, arms, queries, armIdxPair_2_queryIdx, queryIdx_2_armIdxPair, problem_name = problem4_sphere(true, problem_definition_d, problem_definition_K, seed_problem_definition + problem_definition_idx - 1, scale_z)
        problem_config = (θ, arms, queries, armIdxPair_2_queryIdx, queryIdx_2_armIdxPair, problem_name)
        arms_matrix = reduce(vcat, transpose.(arms))'
        @assert rank(arms_matrix) == problem_definition_d
        push!(problem_configs, problem_config)

        design_trans = experimental_design(problem_config, "trans")
        for DDM_barrier_from_0 in DDM_barrier_from_0s
            design_weakPref = experimental_design(problem_config, "weakPref", DDM_barrier_from_0)

            if !haskey(DDMΒarrierFrom0_2_designss, DDM_barrier_from_0)
                DDMΒarrierFrom0_2_designss[DDM_barrier_from_0] = []
            end
            push!(DDMΒarrierFrom0_2_designss[DDM_barrier_from_0], (design_trans, design_weakPref))
        end
    end
    return scale_z, problem_configs, DDMΒarrierFrom0_2_designss
end

@everywhere function run_one_trial(DDM_barrier_from_0::Float64, scale_z::Union{Float64,Int64}, budget::Int64, use_SSM_simulation::Bool, use_SSM_simulation_integration_Δt::Union{Nothing,Float64}, methods::Any, seed_dataset::Int64, repeats_problemDefinition::Int64, repeats_dataset::Int64, problem_configs, designss, base_path::String)
    @assert length(problem_configs) == length(designss)
    println("\nDDM_barrier_from_0=", DDM_barrier_from_0)
    println("scale_z=", scale_z)
    println("budget=", budget)

    nondecision_time = 0.0
    DDM_σ = 1.0
    DDM_α = DDM_barrier_from_0 * 2

    if num_cores == 1
        datas = map(
            ((problem_config_idx, method_idx),) -> run_one_problem(problem_configs[problem_config_idx], designss[problem_config_idx], problem_config_idx, method_idx, seed_dataset, repeats_dataset, budget, nondecision_time, DDM_σ, DDM_α, use_SSM_simulation, use_SSM_simulation_integration_Δt, scale_z, methods),
            Iterators.product(1:repeats_problemDefinition, 1:length(methods))
        )
    else
        datas = pmap(
            ((problem_config_idx, method_idx),) -> run_one_problem(problem_configs[problem_config_idx], designss[problem_config_idx], problem_config_idx, method_idx, seed_dataset, repeats_dataset, budget, nondecision_time, DDM_σ, DDM_α, use_SSM_simulation, use_SSM_simulation_integration_Δt, scale_z, methods),
            Iterators.product(1:repeats_problemDefinition, 1:length(methods))
        )
    end
    @assert size(datas) == (repeats_problemDefinition, length(methods))

    method_2_errorName_2_error = Dict()
    for (method_idx, method) in enumerate(methods)
        error_elims = Float64[]
        num_datas = Int64[]
        for problem_config_idx = 1:repeats_problemDefinition
            data = datas[problem_config_idx, method_idx]
            @assert data["budget"] == budget
            @assert data["problem_config_idx"] == problem_config_idx
            @assert data["method_idx"] == method_idx
            # problem_name = data["problem_name"]

            @assert length(data["error_elims"]) == repeats_dataset
            @assert length(data["num_datas"]) == repeats_dataset

            error_elims = vcat(error_elims, data["error_elims"])
            num_datas = vcat(num_datas, data["num_datas"])
        end

        method_2_errorName_2_error[method] = Dict()
        method_2_errorName_2_error[method]["error_elim"] = (mean(error_elims), std(error_elims), length(error_elims), std(error_elims) / sqrt(length(error_elims)))
        method_2_errorName_2_error[method]["num_data"] = (mean(num_datas), std(num_datas), length(num_datas), std(num_datas) / sqrt(length(num_datas)))
    end
    for (method_idx, method) in enumerate(methods)
        println("\n", method)
        println("num_data=", round.(method_2_errorName_2_error[method]["num_data"]; digits=3))
        println("error_elim=", round.(method_2_errorName_2_error[method]["error_elim"]; digits=3))
    end
    return method_2_errorName_2_error
end

@everywhere function run_one_problem(problem_config::Tuple{Vector{Float64},Vector{Vector{Float64}},Vector{Vector{Float64}},Dict{Tuple{Int64,Int64},Int64},Vector{Tuple{Int64,Int64}},String}, designs::Any, problem_config_idx::Int64, method_idx::Int64, seed_dataset::Int64, repeats_dataset::Int64, budget::Int64, nondecision_time::Float64, DDM_σ::Float64, DDM_α::Float64, use_SSM_simulation::Bool, use_SSM_simulation_integration_Δt::Union{Nothing,Float64}, scale_z::Union{Float64,Int64}, methods::Any)
    (θ, arms, queries, armIdxPair_2_queryIdx, queryIdx_2_armIdxPair, problem_name) = problem_config
    (design_trans, design_weakPref) = designs
    d = length(arms[1])
    K = length(arms)
    M = length(queries)

    phase_η = 2

    (model_name, design_method) = methods[method_idx]
    @assert model_name ∈ ["GLM", "LM"]
    @assert design_method ∈ ["trans", "weakPref"]

    if design_method ∈ ["trans"]
        design = design_trans
    elseif design_method ∈ ["weakPref"]
        design = design_weakPref
    else
        error("Impossible")
    end
    @assert !isnothing(design)

    error_elims = zeros(Float64, repeats_dataset)
    num_datas = zeros(Int64, repeats_dataset)
    for dataset_idx = 1:repeats_dataset
        Random.seed!(seed_dataset + dataset_idx - 1)

        (X, query_idxs, rts, choices_oneTwo, choices_oneZero, choices_oneNegOne, queryIdx_2_rts, queryIdx_2_choiceOneZeros, queryIdx_2_choiceOneNegOnes) = construct_dataset_based_on_design(problem_config, design, budget, nondecision_time, DDM_σ, DDM_α, use_SSM_simulation, use_SSM_simulation_integration_Δt, scale_z, model_name)

        error_elim = nothing
        if model_name ∈ ["GLM"]
            error_elim = compute_GLM(X, choices_oneZero, choices_oneNegOne, query_idxs, queryIdx_2_choiceOneZeros, arms, queries, θ, phase_η, scale_z, nondecision_time, DDM_σ, DDM_α)
        elseif model_name ∈ ["LM"]
            error_elim = compute_LM(X, queryIdx_2_rts, queryIdx_2_choiceOneNegOnes, query_idxs, arms, queries, θ, phase_η, scale_z, nondecision_time, DDM_σ, DDM_α)
        else
            error("Invalid model_name=", model_name)
        end

        error_elims[dataset_idx] = error_elim
        num_datas[dataset_idx] = length(query_idxs)
    end

    ret = Dict{String,Any}()
    ret["error_elims"] = error_elims
    ret["num_datas"] = num_datas

    ret["budget"] = budget
    ret["problem_name"] = problem_name
    ret["problem_config_idx"] = problem_config_idx
    ret["method_idx"] = method_idx
    return ret
end

@everywhere function evaluate_θ_eliminiation_error(θ_hat::Vector{Float64}, arms::Vector{Vector{Float64}}, θ::Vector{Float64}, phase_η::Int64, scale_z::Union{Float64,Int64})
    # 1. Elimination error
    rewards = [arm' * θ for arm in arms]
    armIdx_optimal = argmax(rewards)

    # reward_armIdxs = [(arm' * θ_hat, i) for (i, arm) in enumerate(arms)]
    # reward_armIdxs = shuffle(reward_armIdxs) # random tie breaking for sorting
    # reward_armIdxs = sort(reward_armIdxs, rev=true)
    # new_active = Int64[]
    # middle_idx = ceil(Int64, length(arms) / phase_η)
    # @assert middle_idx >= 1
    # new_active = [x[2] for x in (reward_armIdxs[1:middle_idx])]
    # error = !(armIdx_optimal in new_active)

    # 2. Optimal arm error
    rewards_est = [arm' * θ_hat for arm in arms]
    armIdx_optimal_est = argmax(rewards_est)
    error = (armIdx_optimal_est != armIdx_optimal)
    return error
end

@everywhere function construct_dataset_based_on_design(problem_config::Tuple{Vector{Float64},Vector{Vector{Float64}},Vector{Vector{Float64}},Dict{Tuple{Int64,Int64},Int64},Vector{Tuple{Int64,Int64}},String}, design::Vector{Float64}, budget::Int64, nondecision_time::Float64, DDM_σ::Float64, DDM_α::Float64, use_SSM_simulation::Bool, use_SSM_simulation_integration_Δt::Union{Nothing,Float64}, scale_z::Union{Float64,Int64}, model_name::String)
    if use_SSM_simulation
        @assert !isnothing(use_SSM_simulation_integration_Δt)
    end
    @assert model_name ∈ ["GLM", "LM"]
    (θ, arms, queries, armIdxPair_2_queryIdx, queryIdx_2_armIdxPair, problem_name) = problem_config
    d = length(arms[1])
    K = length(arms)
    M = length(queries)
    design /= sum(design)

    sample_dist = Categorical(design)
    budget_left = budget

    X = zeros(Float64, budget, d)
    query_idxs = zeros(Int64, budget)
    rts = zeros(Float64, budget)
    choices_oneTwo = zeros(Int64, budget)
    cur_idx = 1
    while true
        m_to_sample = rand(sample_dist)

        DDM_dist = DDM(; ν=queries[m_to_sample]' * θ / DDM_σ, α=DDM_α / DDM_σ, τ=nondecision_time, z=0.5)
        if use_SSM_simulation
            Δt = use_SSM_simulation_integration_Δt
            (choice_oneTwo, rt) = simulate_choice_rt(Random.default_rng(), DDM_dist; Δt=Δt)
        else
            (choice_oneTwo, rt) = _rand_rejection(Random.default_rng(), DDM_dist)
        end
        X[cur_idx, :] = queries[m_to_sample]
        query_idxs[cur_idx] = m_to_sample
        choices_oneTwo[cur_idx] = choice_oneTwo
        rts[cur_idx] = rt

        budget_left -= 1
        if budget_left <= 0
            break
        end
        cur_idx += 1
    end
    @assert cur_idx == budget
    @assert all(x -> (x == 0), X[cur_idx+1:end, :])
    @assert all(x -> (x == 0), query_idxs[cur_idx+1:end])
    @assert all(x -> (x == 0), rts[cur_idx+1:end])
    @assert all(x -> (x == 0), choices_oneTwo[cur_idx+1:end])
    X = X[1:cur_idx, :]
    query_idxs = query_idxs[1:cur_idx]
    rts = rts[1:cur_idx]
    choices_oneTwo = choices_oneTwo[1:cur_idx]
    @assert all(x -> (x > nondecision_time), rts)
    @assert all(x -> x ∈ [1, 2], choices_oneTwo)

    choices_oneZero = 2 .- choices_oneTwo
    choices_oneZero = [Bool(x) for x in choices_oneZero]
    choices_oneNegOne = 2 .* choices_oneZero .- 1
    @assert all(x -> x ∈ [1, 0], choices_oneZero)
    @assert all(x -> x ∈ [-1, 1], choices_oneNegOne)

    queryIdx_2_rts = Dict{Int64,Vector{Float64}}()
    queryIdx_2_choiceOneZeros = Dict{Int64,Vector{Bool}}()
    queryIdx_2_choiceOneNegOnes = Dict{Int64,Vector{Int64}}()
    for (i, query_idx) in enumerate(query_idxs)
        if !haskey(queryIdx_2_rts, query_idx)
            queryIdx_2_rts[query_idx] = []
            queryIdx_2_choiceOneZeros[query_idx] = []
            queryIdx_2_choiceOneNegOnes[query_idx] = []
        end
        push!(queryIdx_2_rts[query_idx], rts[i])
        push!(queryIdx_2_choiceOneZeros[query_idx], choices_oneZero[i])
        push!(queryIdx_2_choiceOneNegOnes[query_idx], choices_oneNegOne[i])
    end
    return X, query_idxs, rts, choices_oneTwo, choices_oneZero, choices_oneNegOne, queryIdx_2_rts, queryIdx_2_choiceOneZeros, queryIdx_2_choiceOneNegOnes
end

@everywhere function compute_GLM(X::Matrix{Float64}, choices_oneZero::Vector{Bool}, choices_oneNegOne::Vector{Int64}, query_idxs::Vector{Int64}, queryIdx_2_choiceOneZeros::Dict{Int64,Vector{Bool}}, arms::Vector{Vector{Float64}}, queries::Vector{Vector{Float64}}, θ::Vector{Float64}, phase_η::Int64, scale_z::Union{Float64,Int64}, nondecision_time::Float64, DDM_σ::Float64, DDM_α::Float64)
    # d = length(arms[1])
    # K = length(arms)
    # M = length(queries)

    θ_hat = solve_GLM_in_Algorithms(X, choices_oneZero, choices_oneNegOne, query_idxs, nothing, queries)
    errorElim = evaluate_θ_eliminiation_error(θ_hat, arms, θ, phase_η, scale_z)
    return errorElim
end

@everywhere function compute_LM(X::Matrix{Float64}, queryIdx_2_rts::Dict{Int64,Vector{Float64}}, queryIdx_2_choiceOneNegOnes::Dict{Int64,Vector{Int64}}, query_idxs::Vector{Int64}, arms::Vector{Vector{Float64}}, queries::Vector{Vector{Float64}}, θ::Vector{Float64}, phase_η::Int64, scale_z::Union{Float64,Int64}, nondecision_time::Float64, DDM_σ::Float64, DDM_α::Float64)
    d = length(arms[1])
    # K = length(arms)
    # M = length(queries)

    queryIdx_2_errorEmpMean = Dict{Int64,Float64}()
    queryIdx_2_ERt = Dict{Int64,Float64}()
    queryIdx_2_EChoice = Dict{Int64,Float64}()
    for query_idx in keys(queryIdx_2_rts)
        EChoice = mean(queryIdx_2_choiceOneNegOnes[query_idx])
        ERt = mean(queryIdx_2_rts[query_idx])
        reward = queries[query_idx]' * θ
        queryIdx_2_errorEmpMean[query_idx] = abs(EChoice / ERt - reward)
        queryIdx_2_ERt[query_idx] = ERt
        queryIdx_2_EChoice[query_idx] = EChoice
    end

    Xy = zeros(Float64, d)
    for (i, query_idx) in enumerate(query_idxs)
        EChoice = queryIdx_2_EChoice[query_idx]
        ERt = queryIdx_2_ERt[query_idx]
        Xy += queries[query_idx] .* EChoice / ERt
    end
    θ_hat = nothing
    try
        θ_hat = pinv(X' * X) * Xy
    catch err
        @error "ERROR: solve LM via least squares failed: " err
        θ_hat = qr(X' * X, Val(true)) \ Xy
    end
    errorElim = evaluate_θ_eliminiation_error(θ_hat, arms, θ, phase_η, scale_z)
    return errorElim
end
main()