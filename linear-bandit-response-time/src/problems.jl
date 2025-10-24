struct Problem
    queries::Vector{Vector{Float64}} # array of size M of queries in R^d (X)
    arms::Vector{Vector{Float64}}  # array of size K of arms in R^d (Z)
    queries_matrix::Matrix{Float64} # dxM
    arms_matrix::Matrix{Float64} # dxK
    armIdxPair_2_queryIdx::Dict{Tuple{Int64,Int64},Int64}
    queryIdx_2_armIdxPair::Array{Tuple{Int64,Int64}}
    arms_are_standard_bandits::Bool

    function Problem(queries::Vector{Vector{Float64}}, arms::Vector{Vector{Float64}}, armIdxPair_2_queryIdx::Dict{Tuple{Int64,Int64},Int64}, queryIdx_2_armIdxPair::Array{Tuple{Int64,Int64}})
        @assert length(queries[1]) == length(arms[1])
        queries_matrix = reduce(vcat, transpose.(queries))'
        arms_matrix = reduce(vcat, transpose.(arms))'

        d = length(arms[1])
        arms_are_standard_bandits = true
        for arm in arms
            if count(x -> (x == 0), arm) != (d - 1) || count(x -> (x == 1), arm) != 1
                arms_are_standard_bandits = false
                # println("Linear bandits, due to arm=", round.(arm; digits=3))
                break
            end
        end
        if arms_are_standard_bandits
            # println("Std bandits")
        end

        new(queries, arms, queries_matrix, arms_matrix, armIdxPair_2_queryIdx, queryIdx_2_armIdxPair, arms_are_standard_bandits)
    end
end