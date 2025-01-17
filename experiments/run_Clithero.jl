using Distributed
num_cores = 1
num_cores = 6
if num_cores > 1
    addprocs(num_cores - 1)
end
@everywhere using JLD2, Printf, IterTools, Random, DataStructures
@everywhere using LinearAlgebra, Distributions, GLM
@everywhere using Plots, LaTeXStrings, Measures, Base, Dates
@everywhere using JuMP, Ipopt, YAML
@everywhere include("../src/problems.jl")
@everywhere include("../src/algorithms_REST.jl")
@everywhere include("../src/algorithms_utils.jl")
@everywhere include("../src/runit_REST.jl")
@everywhere include("../src/experiment_helpers.jl")
@everywhere include("./problems.jl")
@everywhere include("../src/GLM.jl")
@everywhere include("../src/DDM_local.jl")

dataset_name = "Clithero"
tune_η = false # To evaluate the performance of the selected best η
# tune_η = true # To determine the best η

function main()
    # https://discourse.julialang.org/t/how-to-set-up-number-of-threads-appropriately-based-on-hardware/63856
    println("# cores=", length(Sys.cpu_info())) # 32

    base_path = @__DIR__
    base_path = base_path * "/"
    println("base_path=", base_path)

    path_parts = splitpath(base_path)
    path_parts[end] = "data"
    SSM_path = joinpath(path_parts...)
    SSM_path = SSM_path * "/" * dataset_name * "_subjectIdx_2_params.jld"
    println("SSM_path=", SSM_path)

    if tune_η
        result_path = base_path * "run_" * dataset_name * "_tune_eta/"
    else
        result_path = base_path * "run_" * dataset_name * "/"
    end
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

    subjectIdx_2_params = load(SSM_path, "subjectIdx_2_params")
    num_subjects = length(subjectIdx_2_params) # 31

    subject_idxs = collect(1:num_subjects)
    # subject_idxs = [1, 2] # for debug

    for subject_idx in subject_idxs
        @time run_1_subject(subject_idx, base_path, SSM_path, result_path, num_subjects)
    end
    process_results_for_Python_plotting(result_path, dataset_name, num_subjects, tune_η)
end

function run_1_subject(subject_idx::Int64, base_path::String, SSM_path::String, result_path::String, num_subjects::Int64)
    println("\n\n------------\nsubject_idx=", subject_idx)
    # https://stackoverflow.com/questions/5012560/how-to-query-seed-used-by-random-random
    # seed = abs.(rand(Int64, num_runs))
    seed_problem_definition = 123
    seed_interaction = 123
    Random.seed!(seed_problem_definition)

    # ---------------------------------------------
    # Buffer size B_buff in Alg.1 in our paper
    budget_buffer_per_phase = 10
    # Elimination parameter η in Alg.1 in our paper
    # Whether benchmark all possible η, or only the η's that result in different number of phases.
    only_ηs_that_have_different_num_phases = false

    if tune_η
        repeats = 50
    else
        repeats = 300
    end
    # repeats = 5 # for debug

    algorithmName_designMethod_phaseηss = nothing
    if tune_η
        # Int64[] means that we don't specify η, and will evalute the performance for all ηs.
        algorithmName_designMethod_phaseηss = [
            ("GLM", "trans", Int64[]), # Choices only + transductive design
            ("LM", "trans", Int64[]), # Our proposed method + transductive design
            ("Chiong24Lemma1", "trans", Int64[]), # Chiong 24 + transductive design (Xiang Chiong, K., Shum, M., Webb, R., & Chen, R. (2024). Combining choice and response time data: A drift-diffusion model of mobile advertisements. Management Science, 70(2), 1238-1257.)
            ("Wagenmakers07Eq5", "trans", Int64[]), # Wagenmakers 07 + transductive design (Wagenmakers, E. J., Van Der Maas, H. L., & Grasman, R. P. (2007). An EZ-diffusion model for response time and accuracy. Psychonomic bulletin & review, 14(1), 3-22.)
            ("GLM", "weakPref", Int64[]), # Choices only + weak preference design (Jun, K. S., Jain, L., Mason, B., & Nassif, H. (2021, July). Improved confidence bounds for the linear logistic model and applications to bandits. In International Conference on Machine Learning (pp. 5148-5157). PMLR.)
        ]
    else
        # We only use the selected best η
        algorithmName_designMethod_phaseηss = [
            ("GLM", "trans", [4]),
            ("LM", "trans", [4]),
            ("Chiong24Lemma1", "trans", [5]),
            ("Wagenmakers07Eq5", "trans", [5]),
            ("GLM", "weakPref", [2]),
        ]
    end

    algorithmName_designMethod_phaseηss = nothing
    if tune_η
        # Int64[] means that we don't specify η, and will evalute the performance for all ηs.
        algorithmName_designMethod_phaseηss = [
            ("GLM", "trans", Int64[]), # Choices only + transductive design
            ("LM", "trans", Int64[]), # Our proposed method + transductive design
            ("Chiong24Lemma1", "trans", Int64[]), # Chiong 24 + transductive design (Xiang Chiong, K., Shum, M., Webb, R., & Chen, R. (2024). Combining choice and response time data: A drift-diffusion model of mobile advertisements. Management Science, 70(2), 1238-1257.)
            ("Wagenmakers07Eq5", "trans", Int64[]), # Wagenmakers 07 + transductive design (Wagenmakers, E. J., Van Der Maas, H. L., & Grasman, R. P. (2007). An EZ-diffusion model for response time and accuracy. Psychonomic bulletin & review, 14(1), 3-22.)
            ("GLM", "weakPref", Int64[]), # Choices only + weak preference design (Jun, K. S., Jain, L., Mason, B., & Nassif, H. (2021, July). Improved confidence bounds for the linear logistic model and applications to bandits. In International Conference on Machine Learning (pp. 5148-5157). PMLR.)
        ]
    else
        # We only use the selected best η
        algorithmName_designMethod_phaseηss = [
            ("GLM", "trans", [9]),
            ("LM", "trans", [6]),
            ("Chiong24Lemma1", "trans", [5]),
            ("Wagenmakers07Eq5", "trans", [9]),
            ("GLM", "weakPref", [9]),
        ]
    end

    budgets = [50, 75, 100, 125, 150, 200, 250, 300]
    # budgets = [100, 200, 300] # for debug

    # ---------------------------------------------
    # https://github.com/JuliaIO/JLD2.jl
    subjectIdx_2_params = load(SSM_path, "subjectIdx_2_params")
    SSM_params = subjectIdx_2_params[subject_idx]
    nondecision_time = SSM_params["τ"]
    DDM_σ = 1.0
    SSM_params["σ"] = DDM_σ
    DDM_barrier_from_0 = SSM_params["α"] / 2

    duel_trans = false
    θ, arms, queries, armIdxPair_2_queryIdx, queryIdx_2_armIdxPair, problem_name = problem7_Clithero_v3(subject_idx, SSM_path, seed_problem_definition, duel_trans)

    @time begin
        run_experiment(queries, arms, armIdxPair_2_queryIdx, queryIdx_2_armIdxPair, θ, problem_name, nondecision_time, DDM_σ, DDM_barrier_from_0, budgets, algorithmName_designMethod_phaseηss, budget_buffer_per_phase, result_path, seed_problem_definition, seed_interaction, repeats, subjectIdx_2_params, subject_idx, only_ηs_that_have_different_num_phases)
    end
end
main()