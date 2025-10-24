import SequentialSamplingModels as SSM
using Plots, CSV, JLD2, DataStructures, Random, Turing, DataFrames
using Statistics, LinearAlgebra, StatsPlots

@model function model_DDM(data::Vector{@NamedTuple{choice::Int64, rt::Float64}}, min_rt::Float64, utility_diffs::Vector{Int64}, use_informative_prior::Bool)
    # https://docs.juliahub.com/General/SequentialSamplingModels/0.9.0/DDM/
    # ν=drift
    # α=barrier_from_0 * 2
    # τ=nondecision_time
    # l=integration speed

    α ~ Uniform(1e-5, 15)
    τ ~ Uniform(1e-3, min_rt)

    # d = 0.0002 ms−1 under Fig.1 in Krajbich, I., Armel, C., & Rangel, A. (2010). Visual fixations and the computation and comparison of value in simple choice. Nature neuroscience, 13(10), 1292-1298.
    l ~ Uniform(1e-10, 1)

    for i in eachindex(data)
        data[i] ~ SSM.DDM(; ν=l * utility_diffs[i], α=α, τ=τ, z=0.5)
    end
end

function main()
    Random.seed!(123)
    base_path = @__DIR__
    base_path = base_path * "/"
    data_path = base_path * "KrajbichDataset.csv"
    file = CSV.File(open(data_path))
    subjectIds = sort!(collect(Set([Int(x) for x in file.subject])))
    num_subjects = length(subjectIds)
    num_trials = length(file.trial)

    subjectId_2_data = Array{Dict}(undef, num_subjects)
    for (ii, subject_idx) in enumerate(subjectIds)
        starting_row_idx = findfirst(x -> (x == subject_idx), file.subject)
        ending_row_idx = findlast(x -> (x == subject_idx), file.subject)
        for i in starting_row_idx:ending_row_idx
            @assert Int(file.subject[i]) == subject_idx
        end
        if starting_row_idx - 1 > 0
            @assert Int(file.subject[starting_row_idx-1]) != subject_idx
        end
        if ending_row_idx + 1 <= num_trials
            @assert Int(file.subject[ending_row_idx+1]) != subject_idx
        end

        choices = file.choice[starting_row_idx:ending_row_idx]
        rts = file.rt[starting_row_idx:ending_row_idx]
        left_ratings = file.leftrating[starting_row_idx:ending_row_idx]
        right_ratings = file.rightrating[starting_row_idx:ending_row_idx]

        subject_data = Dict()
        subject_data["choices"] = [Int64(x) for x in choices]

        # "d is a constant that controls the speed of integration (in units of ms−1)" in Krajbich, I., Armel, C., & Rangel, A. (2010). Visual fixations and the computation and comparison of value in simple choice. Nature neuroscience, 13(10), 1292-1298.
        subject_data["rts"] = rts / 1e3
        subject_data["rts"] = [Float64(x) for x in subject_data["rts"]]
        subject_data["left_ratings"] = left_ratings
        subject_data["right_ratings"] = right_ratings
        subject_data["utility_diffs"] = left_ratings .- right_ratings
        subject_data["subject_idx"] = subject_idx
        subjectId_2_data[ii] = subject_data
    end

    num_data_points = zeros(Int64, length(subjectId_2_data))
    for (subject_idx, subject_data) in enumerate(subjectId_2_data)
        println(subject_idx, ": ", length(subject_data["choices"]))
        num_data_points[subject_idx] = length(subject_data["choices"])
        for k in ["choices", "rts", "left_ratings", "right_ratings", "utility_diffs"]
            @assert num_data_points[subject_idx] == length(subject_data[k])
        end
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

        # List of integers
        utility_diffs = subject_data["utility_diffs"]
        # @NamedTuple{choice::Int64, rt::Float64}[(choice = 2, rt = 2.008005672747187), ..., (choice = 1, rt = 0.3719819194094839)]
        choice_rts = [(choice=subject_data["choices"][i], rt=subject_data["rts"][i]) for i in eachindex(subject_data["choices"])]
        min_rt = minimum(subject_data["rts"])

        @assert all(x -> x.choice ∈ [1, 0], choice_rts) # Note that for DDM training, the coding (1,2) and (1,0) are both ok. Because the package's PDF implementation uses the coding (1, any non-one integers) in https://github.com/itsdfish/SequentialSamplingModels.jl/blob/871d6f8324f1313d3097940d1130c011755fc959/src/DDM.jl#L67.

        # Clithero, J. A. (2018). Improving out-of-sample predictions using response times and a model of the decision process. Journal of Economic Behavior & Organization, 148, 344-375: Using Gibbs sampling, 11000 samples from the posterior were drawn, with the first 1000 discarded as burn-in. As is true for the Logit, the mean of the posterior 10000 samples drawn is used for the out-of-sample DDM predictions.
        # use_informative_prior = true
        use_informative_prior = false
        num_samples = Int(1e4)
        num_chains = 4
        # num_samples = 110 # for debug
        # num_chains = 1 # for debug
        # https://turing.ml/v0.22/docs/using-turing/guide#multithreaded-sampling
        chain = sample(model_DDM(choice_rts, min_rt, utility_diffs, use_informative_prior), NUTS(), MCMCThreads(), num_samples, num_chains)

        summ = summarize(chain[:, :, :], mean, std)
        num_parameters = size(summ)[1]
        for i = 1:num_parameters
            println("paramter=", summ[i, :parameters], ", mean=", round(summ[i, :mean]; digits=3), ", std=", round(summ[i, :std]; digits=3))
        end
        if use_informative_prior
            α = summ[3, :mean]
            τ = summ[6, :mean]
            l = summ[7, :mean]
        else
            α = summ[1, :mean]
            τ = summ[2, :mean]
            l = summ[3, :mean]
        end

        # Check convergence
        # plot(chain; size=(800, 200 * num_parameters))
        # savefig(base_path * "posterior.pdf")
        # exit()

        subjectIdx_2_params[subject_idx]["α"] = α
        subjectIdx_2_params[subject_idx]["τ"] = τ
        subjectIdx_2_params[subject_idx]["l"] = l
        subjectIdx_2_params[subject_idx]["utilities"] = sort(collect(Set(vcat(subject_data["left_ratings"], subject_data["right_ratings"]))))
        subjectIdx_2_params[subject_idx]["νs_in_experiments"] = utility_diffs
    end
    path = base_path * "Krajbich_subjectIdx_2_params.jld"
    println("Saving to ", path)
    # https://github.com/JuliaIO/JLD2.jl
    jldsave(path; subjectIdx_2_params=subjectIdx_2_params)

    # https://csv.juliadata.org/stable/writing.html
    num_utilities_max = maximum([length(v["utilities"]) for (k, v) in subjectIdx_2_params])
    header = vcat(["alpha", "tau", "l intg speed"], ["u" * string(i) for i = 1:num_utilities_max])
    csv_matrix = fill(NaN, length(subjectIdx_2_params), 3 + num_utilities_max)
    for (subject_idx, params) in subjectIdx_2_params
        csv_matrix[subject_idx, 1] = params["α"]
        csv_matrix[subject_idx, 2] = params["τ"]
        csv_matrix[subject_idx, 3] = params["l"]
        for i = 1:num_utilities_max
            if i <= length(params["utilities"])
                csv_matrix[subject_idx, 3+i] = params["utilities"][i]
            else
                csv_matrix[subject_idx, 3+i] = -100
            end
        end
    end
    # println(round.(csv_matrix; digits=3))
    @assert !(any(x -> isnan(x), csv_matrix))
    # https://dataframes.juliadata.org/stable/man/basics/
    tmp = DataFrame(csv_matrix, header)
    path = base_path * "Krajbich_subjectIdx_2_params.csv"
    println("Saving to ", path)
    CSV.write(path, tmp)
end
main()