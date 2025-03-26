using Random, LinearAlgebra, Optim, Plots

# Set parameters
J = 10   # Number of products
M = 5    # Number of characteristics
N = 5000 # Number of consumers

# Generate random product characteristics
Random.seed!(42)
X = randn(J, M)  # Characteristics matrix (J × M)
β_true = rand(M) # True preference parameters

# Generate consumer heterogeneity (random coefficients)
ν = randn(N, M)  # Individual-level taste shocks

# Compute true mean utilities
δ_true = X * β_true  # True mean utilities (J × 1)

# Compute market shares under the mixed logit model
function compute_market_shares(δ, X, ν)
    exp_util = exp.(δ .+ X * ν')  # (J × N)
    sum_exp_util = 1 .+ sum(exp_util, dims=1)  # Include outside option
    return vec(sum(exp_util ./ sum_exp_util, dims=2) / N)  # Mean shares
end

s_true = compute_market_shares(δ_true, X, ν)  # True market shares

# ---- BLP Inversion using Contraction Mapping ----
function blp_inversion(s, X, ν; tol=1e-12, max_iter=1000)
    δ = zeros(J)  # Initial guess
    iter = 0
    for _ in 1:max_iter
        σ = compute_market_shares(δ, X, ν)
        δ_new = δ + log.(s) - log.(σ)  # Contraction mapping update
        iter += 1
        if norm(δ_new - δ) < tol
            println("BLP converged in ", iter, " iterations")
            return δ_new, iter
        end
        δ = δ_new
    end
    println("BLP max iterations reached: ", max_iter)
    return δ, max_iter
end

# ---- KL Divergence Objective for Optim.jl ----
function kl_objective(δ, s, X, ν)
    σ = compute_market_shares(δ, X, ν)
    # Avoid log(0) by clipping σ
    σ = max.(σ, 1e-15)
    kl = sum(s .*  log.(σ))
    return kl
end

function kl_gradient!(G, δ, s, X, ν)
    σ = compute_market_shares(δ, X, ν)
    G[:] = σ - s  # Gradient: σ - s
end

# Run BLP inversion
println("Running BLP Inversion...")
@time begin
    δ_blp, iter_blp = blp_inversion(s_true, X, ν)
end

# Run KL inversion with Optim.jl (L-BFGS solver)
println("\nRunning KL Divergence Inversion with Optim.jl...")
kl_func = δ -> kl_objective(δ, s_true, X, ν)
kl_grad! = (G, δ) -> kl_gradient!(G, δ, s_true, X, ν)
δ0 = ones(J)  # Initial guess
opt = Optim.Options(iterations=100000, f_tol=1e-12, g_tol=1e-12, show_trace=true)
@time begin
    result = optimize(kl_func, δ0, opt)
    δ_kl = Optim.minimizer(result)
    iter_kl = Optim.iterations(result)
end
println("KL converged: ", Optim.converged(result))

# ---- Compare Results ----
println("\nBLP Inversion Error: ", norm(δ_blp - δ_true))
println("KL Divergence Error: ", norm(δ_kl - δ_true))
println("BLP Share Match Error: ", norm(s_true - compute_market_shares(δ_blp, X, ν)))
println("KL Share Match Error: ", norm(s_true - compute_market_shares(δ_kl, X, ν)))

# Plot comparison
plot(1:J, δ_true, label="True δ", marker=:circle, lw=2)
plot!(1:J, δ_blp, label="BLP Inversion", marker=:square, lw=2)
plot!(1:J, δ_kl, label="KL Divergence", marker=:diamond, lw=2)
title!("Comparison of BLP Inversion vs. KL Divergence (Optim.jl)")
xlabel!("Product Index")
ylabel!("Mean Utility (δ)")