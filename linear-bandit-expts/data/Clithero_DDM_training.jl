import SequentialSamplingModels as SSM
using Plots, CSV, JLD2, DataStructures, Random, Turing, DataFrames
using Statistics, LinearAlgebra, StatsPlots, Printf

@model function model_DDM(data::Vector{@NamedTuple{choice::Int64, rt::Float64}}, num_conditions::Union{Nothing,Int64}=nothing, min_rt::Union{Nothing,Float64}=nothing, condition_labels::Union{Nothing,Vector{Int64}}=nothing, use_informative_prior::Bool=false)
    # https://docs.juliahub.com/General/SequentialSamplingModels/0.9.0/DDM/
    # ν=drift
    # α=barrier_from_0 * 2
    # τ=nondecision_time

    if use_informative_prior
        # Informative prior: https://github.com/hddm-devs/hddm/blob/19feaa9849619931d306094b48d9803ef9c8d0bd/hddm/models/hddm_info.py#L430 based on Wiecki, T. V., Sofer, I., & Frank, M. J. (2013). HDDM: Hierarchical Bayesian estimation of the drift-diffusion model in Python. Frontiers in neuroinformatics, 7, 55610.
        μ_α ~ Normal(1.5, sqrt(0.75))
        σ_α ~ truncated(Normal(0.0, sqrt(0.1)), 0.0, Inf)
        α ~ Normal(μ_α, σ_α)
        μ_τ ~ Normal(0.4, sqrt(0.2))
        σ_τ ~ truncated(Normal(0.0, sqrt(1)), 0.0, Inf)
        τ ~ Normal(μ_τ, σ_τ)
        μ_ν ~ Normal(2, sqrt(3))
        σ_ν ~ truncated(Normal(0.0, sqrt(2)), 0.0, Inf)
        νs ~ filldist(Normal(μ_ν, σ_ν), num_conditions)
    else
        # Non-informative prior: "Estimation used non-informative priors: there were no a priori assumptions and all priors were uniform distributions over large intervals of possible parameter values." in Clithero, J. A. (2018). Improving out-of-sample predictions using response times and a model of the decision process. Journal of Economic Behavior & Organization, 148, 344-375.
        α ~ Uniform(1e-5, 15)
        τ ~ Uniform(1e-3, min_rt)
        νs ~ filldist(Uniform(-15, 15), num_conditions)
    end

    for i in eachindex(data)
        ν = νs[condition_labels[i]]
        data[i] ~ SSM.DDM(; ν=ν, α=α, τ=τ, z=0.5)
    end
end

function main()
    Random.seed!(123)
    base_path = @__DIR__
    base_path = base_path * "/"
    data_path = base_path * "ClitheroDataset.csv"
    file = CSV.File(open(data_path))

    num_trials = length(file.trial)
    @assert num_trials == 9486
    subjectIds = sort!(collect(Set([Int(x) for x in file.subj])))
    num_subjects = length(subjectIds)
    @assert num_subjects == 31
    rows_per_subject = Int(num_trials / num_subjects)
    @assert rows_per_subject == 306
    subjectId_2_data = Array{Dict}(undef, num_subjects)
    num_items = 17
    for subject_idx in eachindex(subjectIds)
        rowIdx_range = [(subject_idx - 1) * rows_per_subject + 1, subject_idx * rows_per_subject] # this does not include the header row
        for i in rowIdx_range
            @assert Int(file.subj[i]) == subjectIds[subject_idx]
        end
        if rowIdx_range[1] - 1 > 0
            @assert Int(file.subj[rowIdx_range[1]-1]) != subject_idx
        end
        if rowIdx_range[2] + 1 <= num_trials
            @assert Int(file.subj[rowIdx_range[2]+1]) != subject_idx
        end
        rowIdx_range_2 = [rowIdx_range[1], rowIdx_range[1] + 136 - 1]
        rowIdxs_2 = rowIdx_range_2[1]:rowIdx_range_2[2]
        rowIdx_range_1 = [rowIdx_range[1] + 136, rowIdx_range[2]]
        rowIdxs_1 = rowIdx_range_1[1]:rowIdx_range_1[2]
        @assert file.trial[rowIdx_range_1[1]] == 1
        @assert file.trial[rowIdx_range_2[1]] == 1

        subject_data = Dict()

        # Experiment 1: YN Task
        @assert all(x -> x == 1.0, file.YesNo[rowIdxs_1])
        @assert all(x -> x == 0.0, file.Item_l[rowIdxs_1])
        @assert all(x -> x == 0.0, file.Item_r[rowIdxs_1])
        items_1 = file.Item[rowIdxs_1]
        @assert minimum(items_1) > 0
        choices_1_ = file.response[rowIdxs_1]
        choices_1__ = [Int(x) for x in choices_1_]
        @assert all(x -> x ∈ [0, 1], choices_1__)
        # 1 (yes) => 1, 0 (no) => 2. Note that for DDM training, the coding (1,2) and (1,0) are both ok. Because the package's PDF implementation uses the coding (1, any non-one integers) in https://github.com/itsdfish/SequentialSamplingModels.jl/blob/871d6f8324f1313d3097940d1130c011755fc959/src/DDM.jl#L67.
        choices_1 = [x == 1 ? 1 : 2 for x in choices_1__]
        @assert all(x -> x ∈ [1, 2], choices_1)
        rt_1 = file.rt[rowIdxs_1]
        @assert all(x -> x > 0, rt_1)
        responses_1 = [(choice=choices_1[i], rt=rt_1[i]) for i in eachindex(rowIdxs_1)]
        @assert length(Set(items_1)) == num_items
        @assert length(items_1) == num_items * 10
        subject_data["1_items"] = [Int64(x) for x in items_1]
        subject_data["1_responses"] = responses_1
        items_ref = file.A[rowIdxs_1]

        # Sec.5's footnote 22 in Clithero, J. A. (2018). Improving out-of-sample predictions using response times and a model of the decision process. Journal of Economic Behavior & Organization, 148, 344-375.
        println("--subject_idx=", subject_idx)
        rt_min_outlier = 0.2
        tmp = [x.rt for x in responses_1]
        rt_max_outlier = mean(tmp) + 5 * std(tmp)
        println("outlier RT<0.2: ", count(x -> (x < rt_min_outlier), tmp), " out of ", length(tmp))
        println("outlier RT>", round(rt_max_outlier; digits=3), ": ", count(x -> (x > rt_max_outlier), tmp), " out of ", length(tmp))
        responses_1_trunc = Vector{@NamedTuple{choice::Int64, rt::Float64}}(undef, 0)
        items_1_trunc = Vector{Int64}(undef, 0)
        for (i, x) in enumerate(responses_1)
            if x.rt >= rt_min_outlier && x.rt <= rt_max_outlier
                push!(responses_1_trunc, x)
                push!(items_1_trunc, items_1[i])
            end
        end
        println(length(responses_1), " =>removing outliers=> ", length(responses_1_trunc))
        subject_data["1_items_trunc"] = items_1_trunc
        subject_data["1_responses_trunc"] = responses_1_trunc

        # Experiment 2: dueling Task
        @assert all(x -> x == 0.0, file.YesNo[rowIdxs_2])
        @assert all(x -> x == 0.0, file.Item[rowIdxs_2])
        items_l_2 = file.Item_l[rowIdxs_2]
        @assert minimum(items_l_2) > 0
        items_r_2 = file.Item_r[rowIdxs_2]
        @assert minimum(items_r_2) > 0
        choices_2_ = file.choice[rowIdxs_2]
        choices_2 = [Int(x) for x in choices_2_]
        @assert all(x -> x ∈ [1, 2], choices_2) # Note that for DDM training, the coding (1,2) and (1,0) are both ok. Because the package's PDF implementation uses the coding (1, any non-one integers) in https://github.com/itsdfish/SequentialSamplingModels.jl/blob/871d6f8324f1313d3097940d1130c011755fc959/src/DDM.jl#L67.
        rt_2 = file.rt[rowIdxs_2]
        @assert all(x -> x > 0, rt_2)
        responses_2 = [(choice=choices_2[i], rt=rt_2[i]) for i in eachindex(rowIdxs_2)]
        duels_2 = [(items_l_2[i], items_r_2[i]) for i in eachindex(rowIdxs_2)]
        @assert length(Set(duels_2)) == num_items * (num_items - 1) / 2
        @assert length(duels_2) == num_items * (num_items - 1) / 2
        subject_data["2_duels"] = duels_2
        subject_data["2_responses"] = responses_2

        subjectId_2_data[subject_idx] = subject_data
    end

    num_data_points = zeros(Int64, length(subjectId_2_data))
    for (subject_idx, subject_data) in enumerate(subjectId_2_data)
        num_data_points[subject_idx] = length(subject_data["1_responses_trunc"])
    end
    println("num_data_point ∈ [", minimum(num_data_points), ", ", maximum(num_data_points), "]")

    subject_idxs = 1:num_subjects
    # subject_idxs = 1:2 # for debug

    subjectIdx_2_params = Dict()
    for subject_idx in subject_idxs
        println("---------\nSubject ", subject_idx)
        subject_data = subjectId_2_data[subject_idx]
        subjectIdx_2_params[subject_idx] = Dict()
        subjectIdx_2_params[subject_idx]["raw_data"] = subject_data

        # Training: test_DDM_training.jl
        # List of integers
        items_1 = subject_data["1_items_trunc"]
        # @NamedTuple{choice::Int64, rt::Float64}[(choice = 2, rt = 2.008005672747187), ..., (choice = 1, rt = 0.3719819194094839)]
        responses_1 = subject_data["1_responses_trunc"]

        data_rt = [x.rt for x in responses_1]
        min_rt = minimum(data_rt)
        num_conditions = num_items
        println("length(responses_1)=", length(responses_1))

        @assert all(x -> x.choice ∈ [1, 2], responses_1)

        # Clithero, J. A. (2018). Improving out-of-sample predictions using response times and a model of the decision process. Journal of Economic Behavior & Organization, 148, 344-375: Using Gibbs sampling, 11000 samples from the posterior were drawn, with the first 1000 discarded as burn-in. As is true for the Logit, the mean of the posterior 10000 samples drawn is used for the out-of-sample DDM predictions.
        use_informative_prior = false
        num_samples = Int(1e4)
        num_chains = 4
        # num_samples = 110 # for debug
        # num_chains = 1 # for debug
        # https://turing.ml/v0.22/docs/using-turing/guide#multithreaded-sampling
        chain = sample(model_DDM(responses_1, num_conditions, min_rt, items_1, use_informative_prior), NUTS(), MCMCThreads(), num_samples, num_chains)

        summ = summarize(chain[:, :, :], mean, std)
        num_parameters = size(summ)[1]
        if use_informative_prior
            α = summ[3, :mean]
            τ = summ[6, :mean]
            νs = summ[9:end, :mean]
        else
            α = summ[1, :mean]
            τ = summ[2, :mean]
            νs = summ[3:end, :mean]
        end
        @assert length(νs) == num_items
        for i = 1:num_parameters
            println("paramter=", summ[i, :parameters], ", mean=", round(summ[i, :mean]; digits=3), ", std=", round(summ[i, :std]; digits=3))
        end
        # Check convergence
        # plot(chain; size=(800, 200 * num_parameters))
        # savefig(base_path * "posterior.pdf")
        # exit()

        subjectIdx_2_params[subject_idx]["α"] = α
        subjectIdx_2_params[subject_idx]["τ"] = τ
        subjectIdx_2_params[subject_idx]["νs"] = νs

        utility_is = []
        for i = 1:num_items
            push!(utility_is, (νs[i], i))
        end
        utility_is_ = sort(utility_is, rev=true) # high2low
        tmp = [x[1] for x in utility_is_]
        println("utilities ∈ [", round(minimum(tmp); digits=3), ", ", round(maximum(tmp); digits=3), "]=(", round(mean(tmp); digits=3), ", ", round(std(tmp); digits=3), ")=", round.(tmp; digits=3))
        gap_ijs = []
        for i = 1:num_items
            for j = i+1:num_items
                push!(gap_ijs, (abs(νs[i] - νs[j]), (i, j)))
            end
        end
        gap_ijs_ = sort(gap_ijs, rev=false) # low2high
        gaps = [x[1] for x in gap_ijs_]
        println("gaps ∈ [", round(minimum(gaps); digits=3), ", ", round(maximum(gaps); digits=3), "]=(", round(mean(gaps); digits=3), ", ", round(std(gaps); digits=3), ")=", round.(gaps; digits=3))
    end
    path = base_path * "Clithero_subjectIdx_2_params.jld"
    println("Saving to ", path)
    # https://github.com/JuliaIO/JLD2.jl
    jldsave(path; subjectIdx_2_params=subjectIdx_2_params)

    # https://csv.juliadata.org/stable/writing.html
    header = vcat(["alpha", "tau"], ["nu " * string(i) for i = 1:num_items])
    csv_matrix = fill(NaN, length(subjectIdx_2_params), 2 + num_items)
    for (subject_idx, params) in subjectIdx_2_params
        csv_matrix[subject_idx, 1] = params["α"]
        csv_matrix[subject_idx, 2] = params["τ"]
        for i = 1:num_items
            csv_matrix[subject_idx, 2+i] = params["νs"][i]
        end
    end
    # println(round.(csv_matrix; digits=3))
    @assert !(any(x -> isnan(x), csv_matrix))
    # https://dataframes.juliadata.org/stable/man/basics/
    tmp = DataFrame(csv_matrix, header)
    path = base_path * "Clithero_subjectIdx_2_params.csv"
    println("Saving to ", path)
    CSV.write(path, tmp)
end
main()