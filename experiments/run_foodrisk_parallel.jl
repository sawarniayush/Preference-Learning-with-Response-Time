using Distributed

# Configuration - using a different variable name to avoid conflicts
const DISTRIBUTED_WORKERS = min(Sys.cpu_info() |> length, 6)  # Adjust this number as needed
println("Using $DISTRIBUTED_WORKERS cores")

if DISTRIBUTED_WORKERS > 1
    addprocs(DISTRIBUTED_WORKERS - 1)
    println("Added $(DISTRIBUTED_WORKERS - 1) worker processes, total: $(nprocs()) processes")
end

# Load packages on all workers
@everywhere using JLD2, Printf, IterTools, Random, DataStructures
@everywhere using LinearAlgebra, Distributions, GLM
@everywhere using Plots, LaTeXStrings, Measures, Base, Dates
@everywhere using JuMP, Ipopt, YAML

# Include source files on all workers
@everywhere include("../src/problems.jl")
@everywhere include("../src/algorithms_REST.jl")
@everywhere include("../src/algorithms_utils.jl")
@everywhere include("../src/runit_REST.jl")
@everywhere include("../src/experiment_helpers.jl")
@everywhere include("./problems.jl")
@everywhere include("../src/GLM.jl")
@everywhere include("../src/DDM_local.jl")

dataset_name = "foodrisk"

# Set num_cores = 1 on all workers to force single-threaded operation
# This prevents conflicts with distributed processing
@everywhere begin
    num_cores = 1  # Force single-threaded on each worker
end

# Initialize Python environment on ALL workers (including master)
@everywhere begin
    if isdefined(Main, :learning_functions) == false
        try
            ENV["PYTHON"] = "/home/users/sahasras/.pyenv/shims/python" ###setup your python path here with right packages installed.
            using PyCall
            pushfirst!(PyVector(pyimport("sys")."path"), dirname(@__FILE__) * "/../src")
            global learning_functions = pyimport("learning_functions")
            
            # Test sklearn availability
            sklearn = pyimport("sklearn")
            println("Worker $(myid()): Python environment initialized successfully, sklearn version $(sklearn.__version__)")
        catch e
            @error "Worker $(myid()): Failed to initialize Python environment: $e"
            error("Python initialization failed on worker $(myid())")
        end
    else
        println("Worker $(myid()): Python environment already initialized")
    end
end

println("Python initialized on all workers")

# Define the worker function for processing individual subjects
@everywhere function run_1_subject(subject_idx::Int64, base_path::String, SSM_path::String, result_path::String, num_subjects::Int64)
    worker_id = myid()
    println("Worker $worker_id: Starting subject_idx=$subject_idx")
    
    # Verify Python environment is available on this worker
    if !isdefined(Main, :learning_functions)
        error("Worker $worker_id: learning_functions not available")
    end
    
    seed_problem_definition = 123
    seed_interaction = 123
    Random.seed!(seed_problem_definition)

    # ---------------------------------------------
    # Buffer size B_buff in Alg.1 in our paper
    budget_buffer_per_phase = 20
    if subject_idx ∈ [18]
        budget_buffer_per_phase = 30
    elseif subject_idx ∈ [15]
        budget_buffer_per_phase = 50
    end
    
    # Elimination parameter η in Alg.1 in our paper
    only_ηs_that_have_different_num_phases = false

    ##repeats = 300
    repeats = 70 # for debug

    algorithmName_designMethod_phaseηss = [
        ("GLM", "trans", [2]), # Choices only + transductive design
        ("LM", "trans", [2]), # Previous proposed method + transductive design
        ("LMOrtho", "trans", [2]), # Our proposed orthogonal method + transductive design
        # ("Chiong24Lemma1", "trans", [2]), # Chiong 24 + transductive design
        # ("Wagenmakers07Eq5", "trans", [2]), # Wagenmakers 07 + transductive design
        # ("GLM", "weakPref", [2]), # Choices only + weak preference design
    ]

    ##budgets = [250, 500, 1000, 1500, 2500, 5000]
    budgets = [500, 1000, 1500] # for debug

    # ---------------------------------------------
    # Load subject data
    subjectIdx_2_params = load(SSM_path, "subjectIdx_2_params")
    SSM_params = subjectIdx_2_params[subject_idx]
    nondecision_time = SSM_params["τ"]
    DDM_σ = 1.0
    SSM_params["σ"] = DDM_σ
    DDM_barrier_from_0 = SSM_params["α"] / 2

    θ, arms, queries, armIdxPair_2_queryIdx, queryIdx_2_armIdxPair, problem_name = problem12_foodrisk(subject_idx, SSM_path)

    # Run the experiment for this subject
    @time begin
        result = run_experiment(queries, arms, armIdxPair_2_queryIdx, queryIdx_2_armIdxPair, θ, problem_name, nondecision_time, DDM_σ, DDM_barrier_from_0, budgets, algorithmName_designMethod_phaseηss, budget_buffer_per_phase, result_path, seed_problem_definition, seed_interaction, repeats, subjectIdx_2_params, subject_idx, only_ηs_that_have_different_num_phases)
    end
    
    println("Worker $worker_id: Completed subject_idx=$subject_idx")
    return (subject_idx, :success, result)
end

function main()
    println("# cores=", length(Sys.cpu_info()))
    println("# workers=", nworkers())

    base_path = @__DIR__
    base_path = base_path * "/"
    println("base_path=", base_path)

    path_parts = splitpath(base_path)
    path_parts[end] = "data"
    SSM_path = joinpath(path_parts...)
    SSM_path = SSM_path * "/" * dataset_name * "_subjectIdx_2_params.jld"
    println("SSM_path=", SSM_path)

    result_path = base_path * "run_" * dataset_name * "/"
    if !isdir(result_path)
        mkpath(result_path)
    end

    # Load subject data to get total number of subjects
    subjectIdx_2_params = load(SSM_path, "subjectIdx_2_params")
    num_subjects = length(subjectIdx_2_params) # 42

    subject_idxs = collect(1:num_subjects) ###collect(27:30) # smaller test range to debug

    println("Processing $(length(subject_idxs)) subjects in parallel across $(nworkers()) workers...")
    
    # Process subjects in parallel using pmap
    # pmap automatically distributes work across available workers
    results = pmap(subject_idxs) do subject_idx
        try
            println("Processing subject $subject_idx on worker $(myid())")
            result = run_1_subject(subject_idx, base_path, SSM_path, result_path, num_subjects)
            println("Completed subject $subject_idx on worker $(myid())")
            return result
        catch e
            @error "Error processing subject $subject_idx on worker $(myid()): $e"
            return (subject_idx, :error, e)
        end
    end
    
    # Check results
    successful = filter(r -> r[2] == :success, results)
    failed = filter(r -> r[2] == :error, results)
    
    println("\n" * "="^50)
    println("EXPERIMENT SUMMARY")
    println("="^50)
    println("Total subjects: $(length(subject_idxs))")
    println("Successfully processed: $(length(successful))")
    println("Failed: $(length(failed))")
    
    if !isempty(failed)
        println("\nFailed subjects:")
        for (subject_idx, status, error) in failed
            println("  Subject $subject_idx: $error")
        end
    end
    
    if length(successful) > 0
        println("\nSuccessfully processed subjects: $(sort([r[1] for r in successful]))")
        
        # Run post-processing
        tune_η = false
        println("\nRunning post-processing for Python plotting...")
        process_results_for_Python_plotting(result_path, dataset_name, num_subjects, tune_η)
        println("Post-processing completed!")
    end
    
    # Clean up workers
    println("\nCleaning up workers...")
    rmprocs(workers())
    println("Experiment completed!")
end

# Run the main function
main()