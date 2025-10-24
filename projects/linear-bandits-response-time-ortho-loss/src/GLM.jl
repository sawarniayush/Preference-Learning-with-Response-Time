function project_X_for_GLM(X::Union{Matrix{Float64},Adjoint{Float64,Vector{Float64}}})
    n, d = size(X)
    rank_desired = rank(X)
    if rank_desired < d
        F_ = svd(X' * X; full=false) # dxd
        # U_, Σ_, V_ = F_
        # @assert X'X * V ≈ U * Diagonal(Σ)
        transformation_mat = F_.U[:, 1:rank_desired] # d x rank_desired
        X_ = X * transformation_mat # n x rank_desired
        if rank(X_) != rank_desired
            # Numerical issue. Redo projection with a coarser precision.
            rank_tol = 1e-10
            rank_desired = rank(X, rank_tol)
            transformation_mat = F_.U[:, 1:rank_desired] # d x rank_desired
            X_ = X * transformation_mat # n x rank_desired
            # println(X_)
            # println("rank_desired=", rank_desired, ", d=", d)
            # println("Σ=", F_.S, ", Σ[1:rank_desired]=", F_.S[1:rank_desired])
            @assert rank(X_, rank_tol) == rank_desired
        end
        return X_, transformation_mat
    else
        return X, I
    end
end

function g_probit(t)
    return cdf(Normal(0, 1), t)
end
function logg_probit(t)
    return logcdf(Normal(0, 1), t)
end
function g_prime_probit(t)
    return 1 / sqrt(2 * π) * exp(-0.5 * t^2)
end
function g_logit(t)
    if t > 0
        return 1 / (1 + exp(-t))
    else
        return exp(t) / (1 + exp(t))
    end
end
function logg_logit(t)
    if t > 0
        return -log(1 + exp(-t))
    else
        return t - log(1 + exp(t))
    end
end
function g_prime_logit(t)
    return g_logit(t) * (1 - g_logit(t))
end

function solve_GLM_via_GLM_package(stat_model_name::String, X::Union{Matrix{Float64},Adjoint{Float64,Vector{Float64}}}, y_one_zero::Vector{Bool}, scale_param::Union{Int64,Float64}, verbose::Bool)
    # https://juliastats.org/GLM.jl/stable/api/
    # https://github.com/JuliaStats/GLM.jl/blob/1f0e06bd5daceb8baf96f2b4c3d3babd0599d06a/src/glmfit.jl#L359

    n, d = size(X)
    @assert size(y_one_zero) == (n,)
    @assert stat_model_name ∈ ["probit", "logit"]
    @assert all(x -> x ∈ [0, 1], y_one_zero)

    # scale_param=β for logit, and (sqrt(2) * σ) for probit.
    X = X ./ scale_param
    X, transformation_mat = project_X_for_GLM(X)
    n, d = size(X)
    @assert rank(X) == d

    glm_model = nothing
    if stat_model_name == "logit"
        glm_model = GLM.glm(X, y_one_zero, Bernoulli(), GLM.LogitLink(), dropcollinear=false, verbose=verbose, maxiter=Int(1e5))
    elseif stat_model_name == "probit"
        glm_model = GLM.glm(X, y_one_zero, Bernoulli(), GLM.ProbitLink(), dropcollinear=false, verbose=verbose, maxiter=Int(1e5))
    end
    # https://discourse.julialang.org/t/glm-linear-regression-how-to-extract-the-coefficients/28219
    # https://juliastats.org/GLM.jl/stable/#Methods-applied-to-fitted-models
    θ = GLM.coef(glm_model)
    θ = transformation_mat * θ
    return θ
end

function solve_GLM_via_optimizatiaon(stat_model_name::String, opt_model::Model, X::Union{Matrix{Float64},Adjoint{Float64,Vector{Float64}}}, y_one_negOne::Vector{Int64}, scale_param::Union{Int64,Float64}, regularization::Union{Int64,Float64}=0)
    n, d_ = size(X)
    @assert size(y_one_negOne) == (n,)
    @assert stat_model_name ∈ ["probit", "logit"]
    @assert all(x -> x ∈ [-1, 1], y_one_negOne)

    X = X ./ scale_param

    X, transformation_mat = project_X_for_GLM(X)
    n, d = size(X)
    @assert rank(X) == d

    # Here, we modify the opt_model on the fly, based on test_JuMP.jl.
    empty!(opt_model)

    if stat_model_name == "probit"
        # We ignore the 1/n here
        if d > 1
            function f_probit(l::T...) where {T}
                # Seems like we cannot use vector notation for dot product
                return -sum([logg_probit(sum([y_one_negOne[i] * X[i, j] * l[j] for j = 1:d])) for i = 1:n]) / n + regularization / 2 * sum([ll^2 for ll in l])
            end
            function ∇f_probit(g::AbstractVector{T}, l::T...) where {T}
                λ = zeros(T, n)
                for i = 1:n
                    tmp = sum([y_one_negOne[i] * X[i, j] * l[j] for j = 1:d])
                    λ[i] = g_prime_probit(tmp) / g_probit(tmp) * y_one_negOne[i]
                end
                for j = 1:d
                    g[j] = -sum([λ[i] * X[i, j] for i = 1:n]) / n + regularization * l[j]
                end
                return
            end
            function ∇²f_probit(H::AbstractMatrix{T}, l::T...) where {T}
                λ = zeros(T, n)
                for i = 1:n
                    tmp = sum([y_one_negOne[i] * X[i, j] * l[j] for j = 1:d])
                    λ[i] = g_prime_probit(tmp) / g_probit(tmp) * y_one_negOne[i]
                end
                for i = 1:d
                    for j = i:d
                        # Fill the lower-triangular only.
                        H[j, i] = T(0.0)
                        for k = 1:n
                            xkθ = sum([X[k, p] * l[p] for p = 1:d])
                            H[j, i] += λ[k] * (xkθ + λ[k]) * X[k, i] * X[k, j]
                        end
                        H[j, i] = H[j, i] / n + regularization * (i == j)
                    end
                end
                return
            end
            register(opt_model, :negative_log_likelihood, d, f_probit, ∇f_probit, ∇²f_probit)
            @variable(opt_model, θ[1:d])
            # If you can provide the analytic gradient of the function with respect to your variables then you could use a user-defined operator: Nonlinear Modeling. https://discourse.julialang.org/t/jump-constraint-methoderror/107752
            @NLobjective(opt_model, Min, negative_log_likelihood(θ...))
        else
            function f_probit_1d(l::T) where {T}
                return -sum([logg_probit(y_one_negOne[i] * X[i] * l) for i = 1:n]) / n + regularization / 2 * sum([ll^2 for ll in l])
            end
            function ∇f_probit_1d(l::T) where {T}
                λ = zeros(T, n)
                for i = 1:n
                    tmp = y_one_negOne[i] * X[i] * l
                    λ[i] = g_prime_probit(tmp) / g_probit(tmp) * y_one_negOne[i]
                end
                return -sum([λ[i] * X[i] for i = 1:n]) / n + regularization * l[j]
            end
            function ∇²f_probit_1d(l::T) where {T}
                λ = zeros(T, n)
                for i = 1:n
                    tmp = y_one_negOne[i] * X[i] * l
                    λ[i] = g_prime_probit(tmp) / g_probit(tmp) * y_one_negOne[i]
                end
                return sum([λ[k] * (X[k] * l + λ[k]) * X[k] * X[k] for k = 1:n]) / n + regularization
            end
            register(opt_model, :negative_log_likelihood, 1, f_probit_1d, ∇f_probit_1d, ∇²f_probit_1d)
            @variable(opt_model, θ)
            @NLobjective(opt_model, Min, negative_log_likelihood(θ))
        end

    elseif stat_model_name == "logit"
        if d > 1
            function f_logit(l::T...) where {T}
                return -sum([logg_logit(sum([y_one_negOne[i] * X[i, j] * l[j] for j = 1:d])) for i = 1:n]) / n + regularization / 2 * sum([ll^2 for ll in l])
            end
            function ∇f_logit(g::AbstractVector{T}, l::T...) where {T}
                λ = zeros(T, n)
                for i = 1:n
                    tmp = sum([y_one_negOne[i] * X[i, j] * l[j] for j = 1:d])
                    λ[i] = (g_logit(tmp) - 1) * y_one_negOne[i]
                end
                for j = 1:d
                    g[j] = sum([λ[i] * X[i, j] for i = 1:n]) / n + regularization * l[j]
                end
                return
            end
            function ∇²f_logit(H::AbstractMatrix{T}, l::T...) where {T}
                λ = zeros(T, n)
                for i = 1:n
                    tmp = sum([y_one_negOne[i] * X[i, j] * l[j] for j = 1:d])
                    λ[i] = g_logit(tmp) * (1 - g_logit(tmp))
                end
                for i = 1:d
                    for j = i:d
                        # Fill the lower-triangular only.
                        H[j, i] = sum([λ[k] * X[k, i] * X[k, j] for k = 1:n])
                        H[j, i] = H[j, i] / n + regularization * (i == j)
                    end
                end
                return
            end
            register(opt_model, :negative_log_likelihood, d, f_logit, ∇f_logit, ∇²f_logit)
            @variable(opt_model, θ[1:d])
            @NLobjective(opt_model, Min, negative_log_likelihood(θ...))
        else
            function f_logit_1d(l::T) where {T}
                return -sum([logg_logit(y_one_negOne[i] * X[i] * l) for i = 1:n]) / n + regularization / 2 * sum([ll^2 for ll in l])
            end
            function ∇f_logit_1d(l::T) where {T}
                λ = [(g_logit(y_one_negOne[i] * X[i] * l) - 1) * y_one_negOne[i] for i = 1:n]
                return sum([λ[i] * X[i] for i = 1:n]) / n + regularization * l[j]
            end
            function ∇²f_logit_1d(l::T) where {T}
                λ = zeros(T, n)
                for i = 1:n
                    tmp = y_one_negOne[i] * X[i] * l
                    λ[i] = g_logit(tmp) * (1 - g_logit(tmp))
                end
                return sum([λ[k] * X[k] * X[k] for k = 1:n]) / n + regularization
            end
            register(opt_model, :negative_log_likelihood, d, f_logit_1d, ∇f_logit_1d, ∇²f_logit_1d)
            @variable(opt_model, θ)
            @NLobjective(opt_model, Min, negative_log_likelihood(θ))
        end
    end

    # println(opt_model)
    JuMP.optimize!(opt_model)
    # println(solution_summary(opt_model))
    if termination_status(opt_model) ∉ [LOCALLY_SOLVED, OPTIMAL]
        println(termination_status(opt_model))
        println(solution_summary(opt_model))
        throw(SystemError(string(termination_status(opt_model)), 0))
    end
    @assert primal_status(opt_model) == FEASIBLE_POINT
    # @assert dual_status(opt_model) == FEASIBLE_POINT
    θ_hat = JuMP.value.(opt_model[:θ])
    θ_hat = transformation_mat * θ_hat
    θ_hat = θ_hat[:]
    return θ_hat
end