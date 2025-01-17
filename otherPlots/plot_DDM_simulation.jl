using JLD2, Printf, IterTools, Random, DataStructures
using LinearAlgebra, Distributions
using Plots, LaTeXStrings, Measures, Base
include("../experiments/problems.jl")
include("../src/problems.jl")
include("../src/DDM_local.jl")

color_LM = "#e6550d"
subject_idxs_to_plot = [11]

function main()
    base_path = @__DIR__
    base_path = base_path * "/"
    println("base_path=", base_path)

    path_parts = splitpath(base_path)
    path_parts[end] = "data"
    SSM_path = joinpath(path_parts...)
    SSM_path = SSM_path * "/" * "ClitheroDataset_subjectIdx_2_params.jld"
    println("SSM_path=", SSM_path)

    for subject_idx in subject_idxs_to_plot
        plot_1_subject(subject_idx, base_path, SSM_path)
    end
end

function plot_1_subject(subject_idx::Int64, base_path::String, SSM_path::String)
    println("\n\n------------\nsubject_idx=", subject_idx)
    # https://stackoverflow.com/questions/5012560/how-to-query-seed-used-by-random-random
    seed_problem_definition = 123
    Random.seed!(seed_problem_definition)

    # https://github.com/JuliaIO/JLD2.jl
    subjectIdx_2_params = load(SSM_path, "subjectIdx_2_params")
    SSM_params = subjectIdx_2_params[subject_idx]
    DDM_σ = 1.0
    Δt = 1e-6
    duel_trans = false
    θ, arms, queries, armIdxPair_2_queryIdx, queryIdx_2_armIdxPair, problem_name = problem7_Clithero_v3(subject_idx, SSM_path, seed_problem_definition, duel_trans)

    nondecision_time = SSM_params["τ"]
    # nondecision_time = 0 # plot without t_nondec
    DDM_barrier_from_0 = SSM_params["α"] / 2
    util_diff = mean([q' * θ for q in queries])

    println("nondecision_time=", nondecision_time, ", DDM_barrier_from_0=", DDM_barrier_from_0, ", util_diff=", util_diff)
    DDM_dist = DDM(; ν=util_diff / DDM_σ, α=DDM_barrier_from_0 * 2 / DDM_σ, τ=nondecision_time, z=0.5)

    (time_steps, evidence) = simulate(Random.default_rng(), DDM_dist; Δt=Δt)
    if evidence[end] <= 0
        choice = 2
    elseif evidence[end] >= DDM_dist.α
        choice = 1
    else
        error("Impossible")
    end

    evidence_start_from_y0 = evidence .- DDM_barrier_from_0
    time_steps_start_from_xNondecisionTimes = time_steps .+ nondecision_time
    @assert length(evidence_start_from_y0) == length(time_steps_start_from_xNondecisionTimes)
    subsample_idxs = 1:length(evidence_start_from_y0)

    num_samples = Int(1e3)
    step_size = floor(Int64, length(evidence_start_from_y0) / num_samples)
    @assert step_size > 1
    subsample_idxs = collect(1:step_size:(length(evidence_start_from_y0)+step_size))
    subsample_idxs = [i for i in subsample_idxs if i <= length(evidence_start_from_y0)]

    plot(size=(1000, 700), grid=false) # pixels

    tmp = collect(0:0.01:time_steps_start_from_xNondecisionTimes[1]+0.004)
    plot!(time_steps_start_from_xNondecisionTimes[subsample_idxs], evidence_start_from_y0[subsample_idxs], label="", linewidth=5, color=:red)
    plot!(tmp, [0 for i = eachindex(tmp)], label="", linewidth=5, color=:red)

    # hline!([0], label="", linewidth=1, linestyle=:solid, color=:black)
    hline!([DDM_barrier_from_0], label="", linewidth=5, linestyle=:solid, color=:black)
    hline!([-DDM_barrier_from_0], label="", linewidth=5, linestyle=:solid, color=:black)

    xlabel!("")    # No x-axis label
    # plot!(xlabel=L"Time (sec)")
    plot!(xtickfont=font(30, "Computer Modern"))

    # plot!(ylabel=L"Evidence")
    # Remove y label and y-axis ticks
    ylabel!("")    # No y-axis label
    plot!(yticks=nothing)  # Remove y-axis ticks
    # plot!(ytickfont=font(12, "Computer Modern"))

    # https://discourse.julialang.org/t/removing-numeric-labels-from-axis/89237
    # plot!(ytickfontcolor=:white)
    path = base_path * "plot_DDM_simulation_S" * string(subject_idx)
    path *= ".pdf"
    savefig(path)
    println("Save to ", path)

    rt = time_steps[end] + DDM_dist.τ
    println("choice=", choice, ", rt=", rt)
end
main()