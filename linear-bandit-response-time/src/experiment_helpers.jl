function dump_stats(pep::Problem, algorithm_instances::Vector{Any}, data, repeats::Int64)
    @assert length(algorithm_instances) > 0
    rule = repeat("-", 60)

    @assert size(data) ∈ [(length(algorithm_instances), repeats), (length(algorithm_instances),)]
    alg_names = [data[r, 1][1] for r in eachindex(algorithm_instances)]
    alg_names_max_length = -Inf
    for alg_name in alg_names
        if length(alg_name) > alg_names_max_length
            alg_names_max_length = length(alg_name)
        end
    end
    @assert alg_names_max_length > 0
    alg_names_max_length += 1

    println("")
    println(rule)
    # https://www.tutorialspoint.com/c_standard_library/c_function_sprintf.htm
    # Julia macro: https://www.youtube.com/watch?v=mSgXWpvQEHE&t=579s&ab_channel=TheJuliaProgrammingLanguage
    # https://stackoverflow.com/questions/70119713/how-to-use-function-argument-in-sprintf-in-julia
    prefix_str = x::String -> Printf.format(Printf.Format("%-" * string(alg_names_max_length) * "s"), x)

    results = Array{Dict}(undef, length(algorithm_instances))
    # https://docs.julialang.org/en/v1/base/arrays/#Base.eachindex
    for r in eachindex(algorithm_instances)
        (budget, algorithm_name, design_method, phase_η, nondecision_time, DDM_σ, DDM_barrier_from_0, subtract_nondecision_time) = algorithm_instances[r]

        alg_name = alg_names[r]

        # (algorithm_name_long, mistake, total_cost, total_time, t)
        @assert length([x[1] for x in data[r, :]]) == repeats
        @assert length(Set([x[1] for x in data[r, :]])) == 1
        alg_name_ = collect(Set([x[1] for x in data[r, :]]))[1]
        @assert alg_name == alg_name_

        mistake_at_budget = [x[2] for x in data[r, :]]
        @assert length(mistake_at_budget) == repeats
        mistake_at_budget_mean = mean(mistake_at_budget)
        # https://en.wikipedia.org/wiki/Standard_error
        mistake_at_budget_stderr = std(mistake_at_budget) / sqrt(repeats)
        mistake_at_budget_str = "(" * string(round(mistake_at_budget_mean; digits=2)) * ", " * string(round(mistake_at_budget_stderr; digits=4)) * ")"

        total_cost_at_budget = [x[3] for x in data[r, :]]
        @assert length(total_cost_at_budget) == repeats
        total_cost_at_budget_mean = mean(total_cost_at_budget)
        total_cost_at_budget_stdder = std(total_cost_at_budget) / sqrt(repeats)
        total_cost_at_budget_str = "(" * string(round(total_cost_at_budget_mean; digits=2)) * ", " * string(round(total_cost_at_budget_stdder; digits=4)) * ")"

        num_pulls_at_budget = [x[5] for x in data[r, :]]
        @assert length(num_pulls_at_budget) == repeats
        num_pulls_at_budget_mean = mean(num_pulls_at_budget)
        num_pulls_at_budget_stdder = std(num_pulls_at_budget) / sqrt(repeats)
        num_pulls_at_budget_str = "(" * string(round(num_pulls_at_budget_mean; digits=2)) * ", " * string(round(num_pulls_at_budget_stdder; digits=4)) * ")"

        result = Dict()
        result["alg_name"] = alg_name
        result["budget"] = budget
        result["algorithm_name"] = algorithm_name
        result["design_method"] = design_method
        result["phase_η"] = phase_η
        result["nondecision_time"] = nondecision_time
        result["DDM_σ"] = DDM_σ
        result["DDM_barrier_from_0"] = DDM_barrier_from_0
        result["subtract_nondecision_time"] = subtract_nondecision_time

        result["repeats"] = repeats
        result["mistake_at_budget"] = mistake_at_budget
        result["mistake_at_budget_mean"] = mistake_at_budget_mean
        result["mistake_at_budget_stderr"] = mistake_at_budget_stderr
        result["mistake_at_budget_str"] = mistake_at_budget_str
        result["total_cost_at_budget"] = total_cost_at_budget
        result["total_cost_at_budget_mean"] = total_cost_at_budget_mean
        result["total_cost_at_budget_stdder"] = total_cost_at_budget_stdder
        result["total_cost_at_budget_str"] = total_cost_at_budget_str
        result["num_pulls_at_budget"] = num_pulls_at_budget
        result["num_pulls_at_budget_mean"] = num_pulls_at_budget_mean
        result["num_pulls_at_budget_stdder"] = num_pulls_at_budget_stdder
        result["num_pulls_at_budget_str"] = num_pulls_at_budget_str

        results[r] = result
    end
    println()
    for result in results
        println(
            prefix_str(result["alg_name"]),
            ": num_pulls=", result["num_pulls_at_budget_str"],
            ", total_cost=", result["total_cost_at_budget_str"],
            ", mistake=", result["mistake_at_budget_str"],
            @sprintf(", rep=%d", result["repeats"]),
        )
        println()
    end
    println(rule)
    return results
end


function process_results_for_Python_plotting(result_dir::String, dataset::String, num_subjects::Int64, tune_η::Bool)
    if result_dir[end] != "/"
        result_dir = result_dir * "/"
    end
    tmp = readdir(result_dir)
    tmp = [string(x) for x in tmp if x[end-3:end] == ".dat"]
    if dataset == "foodrisk"
        file_names = [string(x) for x in tmp if x[1:10] == "results_12"]
    elseif dataset == "Clithero"
        file_names = [string(x) for x in tmp if x[1:9] == "results_7"]
    elseif dataset == "Krajbich"
        file_names = [string(x) for x in tmp if x[1:9] == "results_8"]
    else
        error("Invalid dataset=", dataset)
    end

    # s=subject_idx, r=r_desired
    subjectIdx_budget_algorithmName_η_2_result = Dict{Any,Dict{String,Any}}()
    subject_idxs = Int64[]
    budgets = Int64[]
    algorithmName_ηs = Tuple{String,Int64}[]
    algorithm_names = String[]
    ηs = Int64[]

    for file_name in file_names
        tmp = file_name[9:end-4]
        if !isnothing(findfirst("_", tmp))
            subject_idx = parse(Int, tmp[findfirst("s", tmp)[1]+1:findfirst("_", tmp)[1]-1])
        else
            subject_idx = parse(Int, tmp[findfirst("s", tmp)[1]+1:end])
        end
        @assert 0 < subject_idx <= num_subjects
        path = result_dir * file_name
        push!(subject_idxs, subject_idx)

        results = load(path)["results"]

        for result in results
            budget = result["budget"]
            push!(budgets, budget)

            tmp = result["alg_name"]
            alg_name = tmp[1:findfirst("_B", tmp)[1]-1]
            η = result["phase_η"]
            push!(algorithmName_ηs, (alg_name, η))
            push!(algorithm_names, alg_name)
            push!(ηs, η)

            result_dict = Dict()
            for k ∈ [
                "budget",
                "mistake_at_budget_mean", "mistake_at_budget_stderr",
                "repeats"
            ]
                result_dict[k] = result[k]
            end

            kk = (subject_idx, budget, alg_name, η)
            subjectIdx_budget_algorithmName_η_2_result[kk] = result_dict
        end
    end
    subject_idxs = sort(collect(Set(subject_idxs)))
    algorithmName_ηs = sort(collect(Set(algorithmName_ηs)))
    budgets = sort(collect(Set(budgets)))
    algorithm_names = sort(collect(Set(algorithm_names)))
    ηs = sort(collect(Set(ηs)))

    if ~tune_η
        # Save processed results for python
        subject_idxs = sort(collect(Set(subject_idxs)))
        algorithmName_ηs = sort(collect(Set(algorithmName_ηs)))
        budgets = sort(collect(Set(budgets)))
        println("subject_idxs=", subject_idxs)
        println("algorithmName_ηs=", algorithmName_ηs)
        println("budgets=", budgets)

        processed_result = Dict()
        for (alg_name, η) in algorithmName_ηs
            processed_result[alg_name] = Dict()
            processed_result[alg_name][η] = Dict()
            for budget in budgets
                processed_result[alg_name][η][budget] = Dict()
                for subject_idx in subject_idxs
                    kk = (subject_idx, budget, alg_name, η)
                    result = subjectIdx_budget_algorithmName_η_2_result[kk]
                    processed_result[alg_name][η][budget][subject_idx] = result
                end
            end
        end
        path = result_dir * "processed_result.yaml"
        println("\nSave to ", path)
        YAML.write_file(path, processed_result)
    else
        # Save results for determining the best η
        alg_name_2_eta_2_mistakeAtBudgetMeans = Dict()
        alg_name_2_eta_2_vals = Dict()
        for alg_name in algorithm_names
            mistake_at_budget_meanss = []
            alg_name_2_eta_2_mistakeAtBudgetMeans[alg_name] = Dict()
            for η in ηs
                mistake_at_budget_means = Float64[]
                for budget in budgets
                    for subject_idx in subject_idxs
                        kk = (subject_idx, budget, alg_name, η)
                        result = subjectIdx_budget_algorithmName_η_2_result[kk]

                        # It takes too much space, so we don't save it.
                        # rec_mistake_at_budget = result["rec_mistake_at_budget"]

                        num_mistakes = round(Int64, result["mistake_at_budget_mean"] * result["repeats"])
                        @assert abs(num_mistakes - result["mistake_at_budget_mean"] * result["repeats"]) < 1e-5
                        rec_mistake_at_budget = vcat(ones(Int64, num_mistakes), zeros(Int64, result["repeats"] - num_mistakes))
                        # @assert sum(result["rec_mistake_at_budget"]) == num_mistakes

                        # mistake_at_budgets = vcat(mistake_at_budgets, rec_mistake_at_budget)
                        @assert length(rec_mistake_at_budget) == result["repeats"]

                        push!(mistake_at_budget_means, result["mistake_at_budget_mean"])
                    end
                end
                # push!(mistake_at_budgetss, mistake_at_budgets)
                push!(mistake_at_budget_meanss, mistake_at_budget_means)
                alg_name_2_eta_2_mistakeAtBudgetMeans[alg_name][η] = mistake_at_budget_means
            end

            alg_name_2_eta_2_vals[alg_name] = Dict()
            all_results = []
            labels = []
            for (η, mistake_at_budget_means) in zip(ηs, mistake_at_budget_meanss)
                q1 = quantile(mistake_at_budget_means, 0.25) # <q2
                q3 = quantile(mistake_at_budget_means, 0.75) # >q2
                q2 = quantile(mistake_at_budget_means, 0.5)
                @assert q2 == median(mistake_at_budget_means)

                alg_name_2_eta_2_vals[alg_name][η] = [q2, q3, q1]

                append!(all_results, mistake_at_budget_means)
                append!(labels, repeat([η], inner=length(mistake_at_budget_means)))
            end
        end

        println()
        println("alg_name_2_eta_2_vals=", alg_name_2_eta_2_vals)
        alg_name_2_eta = Dict()
        for (alg_name, eta_2_vals) in alg_name_2_eta_2_vals
            # println(alg_name, ": ", eta_2_vals)
            min_eta = nothing
            min_vals = nothing
            for (eta, vals) in eta_2_vals
                if min_vals === nothing || vals < min_vals
                    min_eta = eta
                    min_vals = vals
                end
            end
            alg_name_2_eta[alg_name] = min_eta
        end
        println("alg_name_2_eta=", alg_name_2_eta)
        path = result_dir * "best_etas.yaml"
        YAML.write_file(path, alg_name_2_eta)

        path = result_dir * "alg_name_2_eta_2_mistakeAtBudgetMeans.yaml"
        YAML.write_file(path, alg_name_2_eta_2_mistakeAtBudgetMeans)
    end
end