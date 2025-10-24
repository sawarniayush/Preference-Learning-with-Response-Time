import SequentialSamplingModels as SSM
using Plots, CSV, JLD2, DataStructures, Random, Turing, DataFrames
using Statistics, LinearAlgebra, StatsPlots

@model function model_DDM(data::Vector{@NamedTuple{choice::Int64, rt::Float64}}, num_features::Int64, X::Matrix{Float64}, min_rt::Float64)

    # https://docs.juliahub.com/General/SequentialSamplingModels/0.9.0/DDM/
    # ν=drift
    # α=barrier_from_0 * 2
    # τ=nondecision_time

    α ~ Uniform(1e-5, 15)
    τ ~ Uniform(1e-3, min_rt)
    weights ~ filldist(Uniform(-15, 15), num_features)

    for i in eachindex(data)
        ν = sum([X[i, j] * weights[j] for j = 1:num_features])
        data[i] ~ SSM.DDM(; ν=ν, α=α, τ=τ, z=0.5)
    end
end

function main()
    Random.seed!(123)
    base_path = @__DIR__
    base_path = base_path * "/"
    data_path = base_path * "foodrisk.csv"
    file = CSV.File(open(data_path))

    num_trials = length(file.trial)
    @assert num_trials == 33347

    subjectIds = sort!(collect(Set([Int(x) for x in file.subj])))
    num_subjects = length(subjectIds)
    @assert num_subjects == 42

    subjectId_2_data = Array{Dict}(undef, num_subjects)
    for (ii, subject_idx) in enumerate(subjectIds)
        starting_row_idx = findfirst(x -> (x == subject_idx), file.subj)
        ending_row_idx = findlast(x -> (x == subject_idx), file.subj)
        for i in starting_row_idx:ending_row_idx
            @assert Int(file.subj[i]) == subject_idx
        end
        if starting_row_idx - 1 > 0
            @assert Int(file.subj[starting_row_idx-1]) != subject_idx
        end
        if ending_row_idx + 1 <= num_trials
            @assert Int(file.subj[ending_row_idx+1]) != subject_idx
        end

        trials_ = file.trial[starting_row_idx:ending_row_idx]
        choices_ = file.choice[starting_row_idx:ending_row_idx]
        rts_ = file.rt[starting_row_idx:ending_row_idx]
        upleftval_ = file.upleftval[starting_row_idx:ending_row_idx]
        uprightval_ = file.uprightval[starting_row_idx:ending_row_idx]
        downleftval_ = file.downleftval[starting_row_idx:ending_row_idx]
        downrightval_ = file.downrightval[starting_row_idx:ending_row_idx]

        # Skip the repeats
        trials = Int64[]
        choices = Int64[]
        rts = Float64[]
        upleftval = Int64[]
        uprightval = Int64[]
        downleftval = Int64[]
        downrightval = Int64[]
        for i = 1:length(trials_)
            if trials_[i] ∈ trials
                continue
            end
            push!(trials, trials_[i])
            push!(choices, choices_[i])
            push!(rts, rts_[i])
            push!(upleftval, upleftval_[i])
            push!(uprightval, uprightval_[i])
            push!(downleftval, downleftval_[i])
            push!(downrightval, downrightval_[i])
        end

        subject_data = Dict()
        subject_data["choices"] = [Int64(x) for x in choices]
        subject_data["rts"] = [Float64(x) / 1e3 for x in rts]
        subject_data["upleftval"] = [Float64(x) for x in upleftval] / 10
        subject_data["uprightval"] = [Float64(x) for x in uprightval] / 10
        subject_data["downleftval"] = [Float64(x) for x in downleftval] / 10
        subject_data["downrightval"] = [Float64(x) for x in downrightval] / 10
        subject_data["subject_idx"] = subject_idx
        subjectId_2_data[ii] = subject_data
    end

    num_data_points = zeros(Int64, length(subjectId_2_data))
    for (subject_idx, subject_data) in enumerate(subjectId_2_data)
        num_data_points[subject_idx] = length(subject_data["choices"])
        for k in ["choices", "rts", "upleftval", "uprightval", "downleftval", "downrightval"]
            @assert num_data_points[subject_idx] == length(subject_data[k])
        end
    end
    println("num_data_point ∈ [", minimum(num_data_points), ", ", maximum(num_data_points), "]")

    subject_idxs = 1:num_subjects # This is our indexing, which might be different from subj in the original dataset
    # subject_idxs = 1:1 # for debug

    num_features = 5 # polynomial
    subjectIdx_2_params = Dict()
    for subject_idx in subject_idxs
        println("---------\nSubject ", subject_idx)
        subject_data = subjectId_2_data[subject_idx]
        subjectIdx_2_params[subject_idx] = Dict()
        subjectIdx_2_params[subject_idx]["raw_data"] = subject_data

        # @NamedTuple{choice::Int64, rt::Float64}[(choice = 2, rt = 2.008005672747187), ..., (choice = 1, rt = 0.3719819194094839)]
        choice_rts = [(choice=subject_data["choices"][i], rt=subject_data["rts"][i]) for i in eachindex(subject_data["choices"])]

        X = zeros(Float64, length(subject_data["choices"]), num_features)
        zs = []
        for i in eachindex(subject_data["choices"])
            z1 = [
                subject_data["upleftval"][i],
                subject_data["downleftval"][i],
                (subject_data["upleftval"][i])^2,
                (subject_data["downleftval"][i])^2,
                subject_data["upleftval"][i] * subject_data["downleftval"][i],
            ]
            z2 = [
                subject_data["uprightval"][i],
                subject_data["downrightval"][i],
                (subject_data["uprightval"][i])^2,
                (subject_data["downrightval"][i])^2,
                subject_data["uprightval"][i] * subject_data["downrightval"][i],
            ]
            push!(zs, z1)
            push!(zs, z2)
            X[i, :] = z1 - z2
        end
        zs = collect(Set(zs))
        println("zs=", length(zs), "\n=", zs)

        min_rt = minimum(subject_data["rts"])

        @assert all(x -> x.choice ∈ [1, 0], choice_rts) # Note that for DDM training, the coding (1,2) and (1,0) are both ok. Because the package's PDF implementation uses the coding (1, any non-one integers) in https://github.com/itsdfish/SequentialSamplingModels.jl/blob/871d6f8324f1313d3097940d1130c011755fc959/src/DDM.jl#L67.

        # Clithero, J. A. (2018). Improving out-of-sample predictions using response times and a model of the decision process. Journal of Economic Behavior & Organization, 148, 344-375: Using Gibbs sampling, 11000 samples from the posterior were drawn, with the first 1000 discarded as burn-in. As is true for the Logit, the mean of the posterior 10000 samples drawn is used for the out-of-sample DDM predictions.
        num_samples = Int(1e4)
        num_chains = 4
        # num_samples = 110 # for debug
        # num_chains = 1 # for debug
        # https://turing.ml/v0.22/docs/using-turing/guide#multithreaded-sampling
        chain = sample(model_DDM(choice_rts, num_features, X, min_rt), NUTS(), MCMCThreads(), num_samples, num_chains)

        summ = summarize(chain[:, :, :], mean, std)
        num_parameters = size(summ)[1]
        for i = 1:num_parameters
            println("paramter=", summ[i, :parameters], ", mean=", round(summ[i, :mean]; digits=3), ", std=", round(summ[i, :std]; digits=3))
        end
        @assert num_parameters == 2 + num_features
        α = summ[1, :mean]
        τ = summ[2, :mean]
        θ = summ[3:end, :mean]
        println("α=", α, ", τ=", τ, ", θ=", θ)

        subjectIdx_2_params[subject_idx]["α"] = α
        subjectIdx_2_params[subject_idx]["τ"] = τ
        subjectIdx_2_params[subject_idx]["θ"] = θ
        subjectIdx_2_params[subject_idx]["zs"] = zs

        # Check convergence
        # plot(chain; size=(800, 200 * num_parameters))
        # savefig(base_path * "posterior.pdf")
        # exit()
    end
    path = base_path * "foodrisk_subjectIdx_2_params.jld"
    println("Saving to ", path)
    # https://github.com/JuliaIO/JLD2.jl
    jldsave(path; subjectIdx_2_params=subjectIdx_2_params)

    # https://csv.juliadata.org/stable/writing.html
    header = vcat(["alpha", "tau"], ["th " * string(i) for i = 1:num_features])
    csv_matrix = fill(NaN, length(subjectIdx_2_params), 2 + num_features)
    for (subject_idx, params) in subjectIdx_2_params
        csv_matrix[subject_idx, 1] = params["α"]
        csv_matrix[subject_idx, 2] = params["τ"]
        for i = 1:num_features
            csv_matrix[subject_idx, 2+i] = params["θ"][i]
        end
    end
    # println(round.(csv_matrix; digits=3))
    @assert !(any(x -> isnan(x), csv_matrix))
    # https://dataframes.juliadata.org/stable/man/basics/
    tmp = DataFrame(csv_matrix, header)
    path = base_path * "foodrisk_subjectIdx_2_params.csv"
    println("Saving to ", path)
    CSV.write(path, tmp)
end
main()