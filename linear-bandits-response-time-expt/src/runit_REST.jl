function runIt_RESTAPI(seed::Int64, algorithm_instance::Tuple, pep::Problem, initial_seed::Int64, θ_true::Vector{Float64})
    debug = false
    # debug = true

    # https://discourse.julialang.org/t/random-seed-in-multi-core-parallel-running/41017/2
    Random.seed!(seed)

    start_time = time_ns()
    (budget, algorithm_name, design_method, phase_η, nondecision_time, DDM_σ, DDM_barrier_from_0, subtract_nondecision_time, design_trans_initial_phase, budget_buffer_per_phase) = algorithm_instance

    @assert design_method ∈ ["trans", "weakPref"]
    @assert algorithm_name ∈ ["GLM", "LM", "LMOrthoreuse", "LMOrtho", "Chiong24Lemma1", "Wagenmakers07Eq5"]
    algorithm_name_long = algorithm_name * "_" * design_method
    if !subtract_nondecision_time
        algorithm_name_long *= "_noSubt"
    end
    algorithm_name_long *= "_B" * @sprintf("%.0f", budget) * "_eta" * string(phase_η)

    print_messages = ((initial_seed - seed) % 25 == 0)
    print_messages = false
    if print_messages
        println("Start ", algorithm_name_long, ", seed=", seed)
    end

    d = length(pep.arms[1])
    K = length(pep.arms)
    M = length(pep.queries)

    # [Azizi]=M. Azizi, B. Kveton, and M. Ghavamzadeh. Fixed-budget best-arm identification in structured bandits. In L. D. Raedt, editor, Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI-22, pages 2798–2804.

    # [Azizi]'s Sec.3: ceil(log_η(K))
    num_phases = ceil(Int64, log(K) / log(phase_η)) # ln
    # Burn-in phase without elimination for forming initial estimates. See Jun, K. S., Jain, L., Mason, B., & Nassif, H. (2021, July). Improved confidence bounds for the linear logistic model and applications to bandits. In International Conference on Machine Learning (pp. 5148-5157). PMLR.
    if design_method ∈ ["weakPref"]
        num_phases += 1
    end

    # [Azizi]'s Sec.3: floor(B/s). Here, to allow resource consumption to be non-integer as response times, we just do B/s.
    phase_budgets = ones(num_phases) * (budget / num_phases)
    # Here, since we allow the budget to be non-integer, we don't distribute the left-over budget after flooring to the last several phases (1 per phase) as in https://github.com/Azizimj/StructuredBAI/blob/9eea19175e44bb173f320e9b7f2f2cf0e8999f38/StructBAI.py#L541
    @assert sum(phase_budgets) <= budget + 1e-10
    if debug
        println("phase_budgets=", phase_budgets)
    end

    opt_model = Model(Ipopt.Optimizer)
    set_silent(opt_model)

    max_size_cache = Int(1e4)
    @assert nondecision_time > 0
    num_queries_to_allocate = ceil(Int64, phase_budgets[1] / nondecision_time)

    # Updated at the client
    cur_phase_idx = 0
    cur_num_queries_in_cur_phase = nothing
    cur_choiceOneZero_history = nothing
    cur_rt_history = nothing
    t = 0
    total_cost = 0.0
    # Directly taken from server 1
    cur_active_arms = nothing
    cur_queryArm1Idx_allocation = nothing
    cur_queryArm2Idx_allocation = nothing

    while true
        # 1. Server
        (cur_queryArm1Idx_allocation, cur_queryArm2Idx_allocation, cur_active_arms) = nextSample_RESTAPI(
            num_queries_to_allocate,
            design_trans_initial_phase,
            pep,
            opt_model,
            cur_phase_idx,
            num_phases,
            cur_num_queries_in_cur_phase,
            cur_queryArm1Idx_allocation,
            cur_queryArm2Idx_allocation,
            cur_choiceOneZero_history,
            cur_rt_history,
            cur_active_arms,
            phase_η,
            nondecision_time,
            subtract_nondecision_time,
            algorithm_name,
            design_method,
            debug)
        if debug
            println("\n1. t=", t, ", cur_phase_idx=", cur_phase_idx, ": ", algorithm_name_long)
            println("cur_queryArm1Idx_allocation=", size(cur_queryArm1Idx_allocation), "=", cur_queryArm1Idx_allocation)
            println("cur_queryArm2Idx_allocation=", size(cur_queryArm2Idx_allocation), "=", cur_queryArm2Idx_allocation)
            println("cur_active_arms=", cur_active_arms)
        end

        # 2. Client
        ret = simulate_batch_queries(
            cur_phase_idx, t, total_cost, cur_active_arms,
            cur_queryArm1Idx_allocation, cur_queryArm2Idx_allocation,
            budget_buffer_per_phase,
            pep, θ_true, phase_budgets, num_phases,
            DDM_barrier_from_0, DDM_σ, nondecision_time,
            max_size_cache, algorithm_name_long, debug)
        if length(ret) == 3
            (mistake, t, total_cost) = ret
            total_time = (time_ns() - start_time) / 1e9
            if print_messages
                println("End ", algorithm_name, ": seed=", seed, ", #samples=", t, ", total_cost=", round(total_cost; digits=3), ", mistake=", mistake, ", total_time=", round(total_time; digits=3), "sec\n")
            end
            return (algorithm_name_long, mistake, total_cost, total_time, t)
        elseif length(ret) == 6
            (cur_num_queries_in_cur_phase, cur_choiceOneZero_history, cur_rt_history, cur_phase_idx, t, total_cost) = ret
        else
            error("Impossible")
        end
    end
end

function simulate_batch_queries(
    cur_phase_idx::Int64, t::Int64, total_cost::Float64,
    cur_active_arms::Vector{Int64},
    cur_queryArm1Idx_allocation::Vector{Int64},
    cur_queryArm2Idx_allocation::Vector{Int64},
    budget_buffer_per_phase::Union{Int64,Float64},
    pep::Problem, θ_true::Vector{Float64},
    phase_budgets::Vector{Float64}, num_phases::Int64,
    DDM_barrier_from_0::Float64, DDM_σ::Float64, nondecision_time::Float64,
    max_size_cache::Int64, algorithm_name_long::String, debug::Bool)

    cur_phase_idx += 1

    # 2.1. Check termination
    if cur_phase_idx > num_phases
        @assert length(cur_active_arms) == 1
        rewards = [arm' * θ_true for arm in pep.arms]
        mistake = (cur_active_arms[1] != argmax(rewards))

        return (mistake, t, total_cost)
    end

    cur_num_queries_in_cur_phase = 0
    cur_choiceOneZero_history = zeros(Bool, max_size_cache) # n,
    cur_rt_history = zeros(Float64, max_size_cache) # n,

    cur_budget_remaining = phase_budgets[cur_phase_idx]
    while true
        cur_num_queries_in_cur_phase += 1
        if cur_num_queries_in_cur_phase > min(max_size_cache, length(cur_queryArm1Idx_allocation))
            println("\n4. t=", t, ", cur_phase_idx=", cur_phase_idx, ": ", algorithm_name_long)
            println("cur_num_queries_in_cur_phase=", cur_num_queries_in_cur_phase, ", max_size_cache=", max_size_cache, ", num_queries_to_allocate=", length(cur_queryArm1Idx_allocation))
            error("Insufficient cache in Algorithm.")
        end

        arm1_idx = cur_queryArm1Idx_allocation[cur_num_queries_in_cur_phase]
        arm2_idx = cur_queryArm2Idx_allocation[cur_num_queries_in_cur_phase]

        # test_DDM_local.jl
        util1 = (arm1_idx == 0) ? 0 : pep.arms[arm1_idx]' * θ_true
        util2 = (arm2_idx == 0) ? 0 : pep.arms[arm2_idx]' * θ_true
        util_diff = util1 - util2
        @assert !(arm1_idx == 0 && arm2_idx == 0)

        DDM_dist = DDM(; ν=util_diff / DDM_σ, α=DDM_barrier_from_0 * 2 / DDM_σ, τ=nondecision_time, z=0.5)
        choice, rt = _rand_rejection(Random.default_rng(), DDM_dist)
        choice_oneZero = 2 - choice
        choice_oneNegOne = choice_oneZero * 2 - 1
        @assert rt > 0
        @assert choice ∈ [1, 2]
        @assert choice_oneZero ∈ [1, 0]
        @assert choice_oneNegOne ∈ [-1, 1]

        t += 1
        total_cost += rt
        cur_budget_remaining -= rt
        cur_choiceOneZero_history[cur_num_queries_in_cur_phase] = Bool(choice_oneZero)
        cur_rt_history[cur_num_queries_in_cur_phase] = rt

        if debug
            println("\n2.2. t=", t, ", cur_phase_idx=", cur_phase_idx, ": ", algorithm_name_long)
            println("arm1_idx=", arm1_idx, ", arm2_idx=", arm1_idx, ": choice_oneZero=", choice_oneZero, ", rt=", rt, ", cur_budget_remaining=", cur_budget_remaining)
        end

        if cur_budget_remaining <= budget_buffer_per_phase
            break
        end
    end

    return (cur_num_queries_in_cur_phase, cur_choiceOneZero_history, cur_rt_history, cur_phase_idx, t, total_cost)
end

function run_experiment(queries::Vector{Vector{Float64}}, arms::Vector{Vector{Float64}}, armIdxPair_2_queryIdx::Dict{Tuple{Int64,Int64},Int64}, queryIdx_2_armIdxPair::Array{Tuple{Int64,Int64}}, θ::Vector{Float64}, problem_name::String, nondecision_time::Union{Int64,Float64}, DDM_σ::Union{Int64,Float64}, DDM_barrier_from_0::Union{Int64,Float64}, budgets::Union{Vector{Float64},Vector{Int64}}, algorithmName_designMethod_phaseηss::Vector{Tuple{String,String,Vector{Int64}}}, budget_buffer_per_phase::Union{Int64,Float64}, base_path::String, seed_problem_definition::Int64, seed_interaction::Int64, repeats::Int64, subjectIdx_2_params::Dict{Any,Any}, subject_idx::Int64, only_ηs_that_have_different_num_phases::Bool)
    println()

    d = length(arms[1])
    K = length(arms)
    M = length(queries)

    for phase_η in collect(2:ceil(Int64, K / 2))
        K_cur = K
        print("phase_η=", phase_η, ": ", K_cur)
        while true
            K_new = ceil(Int64, K_cur / phase_η)
            print(" => ", K_new)
            K_cur = K_new
            if K_cur == 1
                break
            end
        end
        println()
    end

    all_phase_ηs = collect(2:ceil(Int64, K / 2))
    if only_ηs_that_have_different_num_phases
        all_phase_ηs = Int64[]
        for num_phases = 2:K
            phase_η = ceil(Int64, exp(log(K) / num_phases))
            num_phases2 = ceil(Int64, log(K) / log(phase_η)) # ln
            println("phase_η=", phase_η, " => num_phases=", num_phases2)
            push!(all_phase_ηs, phase_η)
        end
    end
    all_phase_ηs = sort(collect(Set(all_phase_ηs)))
    println("all_phase_ηs=", all_phase_ηs, " => num_phases=", [ceil(Int64, log(K) / log(phase_η)) for phase_η in all_phase_ηs])

    pep = Problem(queries, arms, armIdxPair_2_queryIdx, queryIdx_2_armIdxPair)

    problem_config = (θ, arms, queries, armIdxPair_2_queryIdx, queryIdx_2_armIdxPair, problem_name)
    design_trans_initial_phase = experimental_design(problem_config, "trans")

    algorithm_instances = []
    for budget in budgets
        for (algorithm_name, design_method, phase_ηs_) ∈ algorithmName_designMethod_phaseηss
            if length(phase_ηs_) == 0
                phase_ηs = all_phase_ηs
            else
                phase_ηs = phase_ηs_
            end
            for phase_η in phase_ηs
                subtract_nondecision_times = [true]
                if algorithm_name ∈ ["LM"]
                    subtract_nondecision_times = [true, false]
                end
                for subtract_nondecision_time in subtract_nondecision_times
                    tmp = (budget, algorithm_name, design_method, phase_η, nondecision_time, DDM_σ, DDM_barrier_from_0, subtract_nondecision_time, design_trans_initial_phase, budget_buffer_per_phase)
                    push!(algorithm_instances, tmp)
                end
            end
        end
    end
    @assert length(algorithm_instances) > 0

    if num_cores > 1
        data = pmap(
            ((algorithm_instance, i),) -> runIt_RESTAPI(seed_interaction + i, algorithm_instance, pep, seed_interaction, θ),
            Iterators.product(algorithm_instances, 1:repeats),
        )
    else
        data = map(
            ((algorithm_instance, i),) -> runIt_RESTAPI(seed_interaction + i, algorithm_instance, pep, seed_interaction, θ),
            Iterators.product(algorithm_instances, 1:repeats),
        )
    end
    results = dump_stats(pep, algorithm_instances, data, repeats)

    path = base_path * "results_" * problem_name * ".dat"
    @save path seed_problem_definition seed_interaction algorithmName_designMethod_phaseηss repeats subjectIdx_2_params subject_idx θ arms queries results
end