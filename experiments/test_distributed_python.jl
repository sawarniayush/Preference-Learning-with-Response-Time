#!/usr/bin/env julia

# Simple test for Distributed + Python (completely top-level)
using Distributed

println("=== Simple Distributed + Python Test ===")
println("Julia version: ", VERSION)
println("Starting with $(nprocs()) process(es)")

# Add workers
if nprocs() == 1
    println("Adding 2 workers...")
    addprocs(2)
    println("Now have $(nprocs()) processes ($(nworkers()) workers)")
else
    println("Already have $(nworkers()) workers")
end

# Import PyCall on ALL workers at top level
@everywhere begin
    ENV["PYTHON"] = "/home/users/sahasras/.pyenv/shims/python"
    using PyCall
end

println("PyCall imported on all workers")

# Define test functions on all workers
@everywhere begin
    function test_basic_python()
        worker_id = myid()
        println("Worker $worker_id: Testing basic Python...")
        
        try
            np = pyimport("numpy")
            sys = pyimport("sys")
            
            x = np.array([1, 2, 3, 4, 5])
            mean_val = np.mean(x)
            std_val = np.std(x)
            result = mean_val * 2 + std_val
            
            return (
                worker_id = worker_id,
                status = :success,
                python_version = split(sys.version)[1],
                numpy_version = np.__version__,
                mean_val = Float64(mean_val),
                std_val = Float64(std_val),
                result = Float64(result)
            )
        catch e
            return (
                worker_id = worker_id,
                status = :error,
                error = string(e)
            )
        end
    end

    function test_sklearn_simple()
        worker_id = myid()
        println("Worker $worker_id: Testing sklearn...")
        
        try
            sklearn = pyimport("sklearn")
            LinearRegression = pyimport("sklearn.linear_model").LinearRegression
            np = pyimport("numpy")
            
            X = np.reshape(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), (-1, 1))
            y = np.array([2.1, 3.9, 6.1, 7.9, 10.1])
            
            model = LinearRegression()
            model.fit(X, y)
            
            score = model.score(X, y)
            coef = model.coef_[1]
            intercept = model.intercept_
            
            return (
                worker_id = worker_id,
                status = :success,
                sklearn_version = sklearn.__version__,
                r2_score = Float64(score),
                coefficient = Float64(coef),
                intercept = Float64(intercept)
            )
        catch e
            return (
                worker_id = worker_id,
                status = :error,
                error = string(e)
            )
        end
    end

    function parallel_computation_task(task_id)
        worker_id = myid()
        
        try
            np = pyimport("numpy")
            np.random.seed(task_id * 42)
            
            data = np.random.randn(100)
            # Use np.square instead of data * data to avoid the multiplication error
            sum_squares = np.sum(np.square(data))
            mean_val = np.mean(data)
            
            return (
                worker_id = worker_id,
                task_id = task_id,
                status = :success,
                sum_squares = Float64(sum_squares),
                mean_val = Float64(mean_val)
            )
        catch e
            return (
                worker_id = worker_id,
                task_id = task_id,
                status = :error,
                error = string(e)
            )
        end
    end

    function test_learning_functions_path()
        worker_id = myid()
        
        try
            # Use the EXACT same approach as your original code
            if isdefined(Main, :learning_functions) == false
                sys = pyimport("sys")
                
                # Try the exact path format from your original code
                src_path = dirname(@__FILE__) * "/../src"
                
                # Check if path exists
                if !isdir(src_path)
                    return (
                        worker_id = worker_id,
                        status = :error,
                        error = "Source directory does not exist: $src_path"
                    )
                end
                
                # List contents
                src_contents = readdir(src_path)
                
                # Use pushfirst! exactly like your original code
                pushfirst!(PyVector(pyimport("sys")."path"), src_path)
                
                # Import exactly like your original code
                global learning_functions = pyimport("learning_functions")
                
                return (
                    worker_id = worker_id,
                    status = :success,
                    src_path = src_path,
                    src_contents = src_contents,
                    module_loaded = isdefined(Main, :learning_functions)
                )
            else
                return (
                    worker_id = worker_id,
                    status = :success,
                    src_path = "already_loaded",
                    src_contents = ["already_loaded"],
                    module_loaded = true
                )
            end
            
        catch e
            # Get more detailed error info
            return (
                worker_id = worker_id,
                status = :error,
                error = string(e),
                error_type = typeof(e)
            )
        end
    end
end

# Test 1: Basic Python functionality
println("\n--- Test 1: Basic Python functionality ---")
basic_futures = [@spawnat w test_basic_python() for w in workers()]
basic_results = [fetch(f) for f in basic_futures]

println("Basic Python test results:")
basic_success_count = 0
for result in basic_results
    if result.status == :success
        global basic_success_count += 1
        println("✓ Worker $(result.worker_id): SUCCESS")
        println("  Python $(result.python_version), NumPy $(result.numpy_version)")
        println("  Mean: $(result.mean_val), Std: $(result.std_val), Result: $(result.result)")
    else
        println("✗ Worker $(result.worker_id): FAILED")
        println("  Error: $(result.error)")
    end
end

if basic_success_count == 0
    println("❌ Basic Python test failed on all workers. Cannot proceed.")
    rmprocs(workers())
    exit(1)
end

# Test 2: Sklearn functionality
println("\n--- Test 2: Sklearn functionality ---")
sklearn_futures = [@spawnat w test_sklearn_simple() for w in workers()]
sklearn_results = [fetch(f) for f in sklearn_futures]

println("Sklearn test results:")
sklearn_success_count = 0
for result in sklearn_results
    if result.status == :success
        global sklearn_success_count += 1
        println("✓ Worker $(result.worker_id): sklearn $(result.sklearn_version) SUCCESS")
        println("  R² score: $(round(result.r2_score, digits=4))")
        println("  Coefficient: $(round(result.coefficient, digits=3)) (should be ~2)")
        println("  Intercept: $(round(result.intercept, digits=3))")
    else
        println("✗ Worker $(result.worker_id): sklearn FAILED")
        println("  Error: $(result.error)")
    end
end

# Test 3: Parallel processing
println("\n--- Test 3: Parallel processing ---")
tasks = 1:6
println("Running $(length(tasks)) parallel tasks...")
println("Available workers: $(workers())")

# Try a simpler approach first
parallel_results = []
for (i, task_id) in enumerate(tasks)
    worker_id = workers()[((i-1) % nworkers()) + 1]
    println("Assigning task $task_id to worker $worker_id")
    try
        future = @spawnat worker_id parallel_computation_task(task_id)
        result = fetch(future)
        push!(parallel_results, result)
        println("Task $task_id completed successfully")
    catch e
        println("Task $task_id failed: $e")
        push!(parallel_results, (
            worker_id = worker_id,
            task_id = task_id,
            status = :error,
            error = string(e)
        ))
    end
end

println("Parallel processing results:")
parallel_success_count = 0
for result in parallel_results
    if result.status == :success
        global parallel_success_count += 1
        println("✓ Task $(result.task_id) on Worker $(result.worker_id): SUCCESS")
        println("  Sum of squares: $(round(result.sum_squares, digits=2))")
        println("  Mean: $(round(result.mean_val, digits=3))")
    else
        println("✗ Task $(result.task_id) on Worker $(result.worker_id): FAILED")
        println("  Error: $(result.error)")
    end
end

# Test 4: Learning functions path
println("\n--- Test 4: Testing learning_functions path ---")
lf_futures = [@spawnat w test_learning_functions_path() for w in workers()]
lf_results = [fetch(f) for f in lf_futures]

println("Learning functions path test results:")
lf_success_count = 0
for result in lf_results
    if result.status == :success
        global lf_success_count += 1
        println("✓ Worker $(result.worker_id): learning_functions module loaded successfully")
        println("  Source path: $(result.src_path)")
        if haskey(result, :src_contents)
            println("  Contents: $(result.src_contents)")
        end
        println("  Module loaded: $(result.module_loaded)")
    else
        println("✗ Worker $(result.worker_id): learning_functions FAILED")
        println("  Error: $(result.error)")
        if haskey(result, :error_type)
            println("  Error type: $(result.error_type)")
        end
        if haskey(result, :src_contents)
            println("  Directory contents: $(result.src_contents)")
        end
    end
end

# Final summary
println("\n" * "="^50)
println("FINAL SUMMARY")
println("="^50)
println("Workers: $(nworkers())")
println("Basic Python tests passed: $basic_success_count/$(nworkers())")
println("Sklearn tests passed: $sklearn_success_count/$(nworkers())")
println("Parallel tasks completed: $parallel_success_count/$(length(tasks))")
println("Learning functions import: $lf_success_count/$(nworkers())")

if basic_success_count == nworkers() && sklearn_success_count == nworkers() && parallel_success_count == length(tasks) && lf_success_count == nworkers()
    println("\n🎉 PERFECT! All tests passed!")
    println("✅ Distributed processing works")
    println("✅ Python integration works")
    println("✅ Sklearn works")
    println("✅ Parallel execution works")
    println("✅ Learning functions module accessible")
    println("\nYou're ready to use distributed processing with your learning_functions!")
else
    println("\n⚠️ Test results:")
    if basic_success_count < nworkers()
        println("❌ Basic Python setup issues")
    end
    if sklearn_success_count < nworkers()
        println("❌ Sklearn issues")
    end
    if parallel_success_count < length(tasks)
        println("❌ Parallel processing issues")
    end
    if lf_success_count < nworkers()
        println("❌ Learning functions import issues")
        println("   This might be fixable - check the diagnostic info above")
    end
    
    if basic_success_count == nworkers() && sklearn_success_count == nworkers() && parallel_success_count == length(tasks)
        println("✅ Basic distributed Python setup works!")
        println("➡️ Only the learning_functions import needs to be fixed")
    end
end

# Clean up
println("\nCleaning up workers...")
rmprocs(workers())
println("Test completed.")