using Distributions:Random
using AdaptiveABC
using SequentialLOB
using Random
using Distributions
using Optim
using NLopt
using JLD2
using Plots

# Set Simulator Parameters and Seed
D = 1.0
σ = 1.5
nu = 0.5
μ = 1.0
α_slob = 100.0
parameter_true = [D, σ, nu, μ, α_slob]
parameter_names = ["D", "σ", "nu", "μ", "α"]
n_replications = 10
Random.seed!(19392910901)
# Constants
num_paths = 1; M = 400 ; T = 2299 ; p₀ = 238.75 ; L = 200
Δx = L / M ; λ = 1.0 ; Δt = (Δx^2) / (2.0 * D) ; 
println("Δx = $Δx and Δt = $Δt")



# Create model, simulate to obtain observation vector and then calculate summary statistics
true_slob_model = SLOB(num_paths,
    T, p₀, M, L, D, σ, nu, α_slob, SourceTerm(λ, μ))
true_slob_price_path = true_slob_model(1921991)
true_slob_log_returns = diff(log.(true_slob_price_path[:,1]))
true_slob_summary_stats = get_summary_stats(true_slob_log_returns)

# plt = plot(
#     1:T + 1, true_slob_price_path[:,1], legend=false,
#     xlab="Time", ylab="Mid-Price",
# )

# Define Summary function used in AdaptiveABC. Takes in a vector of parameters and outputs a success boolean and the 
# simulated summart statistics
function summary_fn(parameters, n_summary_stats, n_replications)
    try
        D, σ, nu, μ, α_slob = parameters
        model = SLOB(num_paths, T, p₀, M, L, D, σ, nu, α_slob, SourceTerm(λ, μ))
        summary_stats = Array{Float64,2}(undef, n_summary_stats, n_replications)
        for i in 1:n_replications
            sim_price_path = model()
            sim_obs = diff(log.(sim_price_path[:, 1]))
            summary_stats[:, i] = get_summary_stats(true_slob_log_returns, sim_obs)    
        end
        return true, summary_stats
    catch e
        return false, zeros(n_summary_stats, n_replications)
    end
end

# Setup for Calibration of Parameters through AdaptiveABC
# Define Prior distribution using Prior struct
prior = Prior([
    Uniform(0.5, 3.0), # D
    Uniform(0.1, 3.0), # σ
    Uniform(0.05, 0.95), # nu
    Uniform(0.1, 3.0), # μ
    Uniform(50.0, 200.0) # α_slob
])

#####
# Calibration Technique: ABC Rejection BBWM
#####

# Distance Function to measure distance between observed summary stats and simulated summary stats
boot_weight_matrix = BlockBootstrapWeightMatrix(9201292, vcat([0], true_slob_log_returns), get_summary_stats, 100, 10_000)
weighted_bootstrap = WeightedBootstrap(vcat([0], true_slob_log_returns), get_summary_stats, boot_weight_matrix)



abc_input_wb = ABCInput(
    prior,
    parameter_names,
    true_slob_summary_stats,
    summary_fn,
    17,
    weighted_bootstrap
)

abc_reject_wb = ABCRejection(
    abc_input_wb,
    3_000,
    n_replications,
    100.0,
    parallel=true,
    seed=182717
)

posterior_means = AdaptiveABC.parameter_means(abc_reject_wb)
println()
println("ABC rejection BBWM: D, σ, nu, μ, α")
print(round.(posterior_means[:,end], digits=3))
println()

JLD2.save_object("$(dirname(@__FILE__))/abc_reject_wb.jld2", abc_reject_wb)

# ABC rejection BBWM: D, σ, nu, μ, α
# [1.755, 1.865, 0.696, 1.538, 13.738]

# param_inds = [1,2,3,4,5]
# plt = plot(abc_reject_wb, iteration_colours=cgrad(:blues, 5, categorical=true),
#     iterations=[1], params_true=parameter_true[param_inds], 
#     prior_dists=abc_input_wb.prior.distribution[param_inds], 
#     param_inds=param_inds, param_names=parameter_names[param_inds])

# savefig(plt, "$(dirname(@__FILE__))/abc_reject_wb.pdf")


#####
# Calibration Technique: ABC Rejection MADWE
#####
abc_input_we = ABCInput(
    prior,
    parameter_names,
    true_slob_summary_stats,
    summary_fn,
    17,
    WeightedEuclidean(true_slob_log_returns, get_summary_stats, ones(17), "MAD")
)

abc_reject_we = ABCRejection(
    abc_input_we,
    3_000,
    n_replications,
    100.0,
    parallel=true,
    seed=1326110
)
posterior_means = AdaptiveABC.parameter_means(abc_reject_we)
println()
println("ABC rejection MADWE: D, σ, nu, μ, α")
print(round.(posterior_means[:,end], digits=3))
println()

JLD2.save_object("$(dirname(@__FILE__))/abc_reject_we.jld2", abc_reject_we)

# ABC rejection MADWE: D, σ, nu, μ, α
# [1.704, 1.702, 0.632, 1.613, 12.52]

# param_inds = [1,2,3,4,5]
# plt = plot(abc_reject_we, iteration_colours=cgrad(:blues, 5, categorical=true),
#     iterations=[1], params_true=parameter_true[param_inds], 
#     prior_dists=abc_input_we.prior.distribution[param_inds], 
#     param_inds=param_inds, param_names=parameter_names[param_inds])

# savefig(plt, "$(dirname(@__FILE__))/abc_reject_we.pdf")

#####
# Calibration Technique: ABC-PMC BBWM
#####


# Distance Function to measure distance between observed summary stats and simulated summary stats
boot_weight_matrix = BlockBootstrapWeightMatrix(9201292, vcat([0], true_slob_log_returns), get_summary_stats, 100, 10_000)
weighted_bootstrap = WeightedBootstrap(vcat([0], true_slob_log_returns), get_summary_stats, boot_weight_matrix)

abc_input_wb = ABCInput(
    prior,
    parameter_names,
    true_slob_summary_stats,
    summary_fn,
    17,
    weighted_bootstrap
)

abc_pmc_out_wb = ABC_PMC(
    abc_input_wb,
    100,
    n_replications,
    0.25,
    10_000,
    10;
    parallel=true,
    batch_size=250,
    seed=718731899
)
posterior_means = AdaptiveABC.parameter_means(abc_pmc_out_wb)

println()
println("ABC-PMC BBWM: D, σ, nu, μ, α_slob")
print(round.(posterior_means[:,end], digits=3))
println()

JLD2.save_object("$(dirname(@__FILE__))/abc_pmc_out_wb.jld2", abc_pmc_out_wb)

# ABC-PMC BBWM: D, σ, nu, μ, α_slob 
# [1.957, 1.973, 0.688, 1.648, 13.338]

# param_inds = [1,2,3,4,5]
# plt = plot(abc_pmc_out_wb, iteration_colours=cgrad(:blues, 5, categorical=true),
#     iterations=[1,5,10], params_true=parameter_true[param_inds], 
#     prior_dists=abc_input_wb.prior.distribution[param_inds], 
#     param_inds=param_inds, param_names=parameter_names[param_inds])
# plt
# savefig(plt, "$(dirname(@__FILE__))/abc_pmc_wb.pdf")  


#####
# Calibration Technique: ABC-PMC MADWE
#####
abc_input_we = ABCInput(
    prior,
    parameter_names,
    true_slob_summary_stats,
    summary_fn,
    17,
    WeightedEuclidean(true_slob_log_returns, get_summary_stats, ones(17), "MAD")
)

abc_pmc_out_we = ABC_PMC(
    abc_input_we,
    100,
    n_replications,
    0.15,
    10_000,
    10;
    parallel=true,
    batch_size=250,
    seed=77261511
)
posterior_means = AdaptiveABC.parameter_means(abc_pmc_out_we)
println("")
println("ABC-PMC MADWE: D, σ, nu, μ, α")
print(round.(posterior_means[:,end], digits=3))
println("")
# ABC-PMC MADWE: D, σ, nu, μ, α_slob
# [1.79, 1.837, 0.693, 1.561, 13.391]

JLD2.save_object("$(dirname(@__FILE__))/abc_pmc_out_we.jld2", abc_pmc_out_we)

# param_inds = [1,2,3,4,5]
# plt = plot(abc_pmc_out_we, iteration_colours=cgrad(:blues, 5, categorical=true),
#     iterations=[1,abc_pmc_out_we.n_iterations], params_true=parameter_true[param_inds], 
#     prior_dists=abc_input_we.prior.distribution[param_inds], 
#     param_inds=param_inds, param_names=parameter_names[param_inds])
# savefig(plt, "$(dirname(@__FILE__))/abc_pmc_we.pdf")  
# plt

#####
# Calibration Technique: Nelder-Mead BBWM
#####

# Distance Function to measure distance between observed summary stats and simulated summary stats
boot_weight_matrix = BlockBootstrapWeightMatrix(9201292, vcat([0], true_slob_log_returns), get_summary_stats, 100, 10_000)
weighted_bootstrap = WeightedBootstrap(vcat([0], true_slob_log_returns), get_summary_stats, boot_weight_matrix)

function f(x, weight, reps)
    success, sum_stats = summary_fn(x, 17, reps)
    if success == true
        return weight(sum_stats)
    else
        return Inf
    end
    
end

f_wb(x, grad) = f(x, weighted_bootstrap, n_replications)
# D, σ, nu, μ, α
lower = [0.5, 0.1, 0.05, 0.1, 50.0]
upper = [3.0, 3.0, 0.95, 3.0, 200.0]

nm_reps = 10
results = zeros(size(lower, 1), nm_reps)
Random.seed!(13241091)
for i in 1:nm_reps
    println("Iteration $i")
    opt = Opt(:LN_NELDERMEAD, size(lower, 1))
    opt.lower_bounds = lower
    opt.upper_bounds = upper
    opt.xtol_rel = 1e-6
    opt.min_objective = f_wb
    init_params = rand(Int.(rand(MersenneTwister(), UInt32)), prior)

    (minf, minx, ret) = NLopt.optimize(opt, init_params)

    results[:,i] = minx
end
println()
println("Nelder-Mead BBWM: ")
print(round.(mean(results, dims=2), digits=3))
print()
results
println()
# Nelder-Mead BBWM: 
# [1.735; 1.775; 0.599; 2.203; 125.624]

#####
# Calibration Technique: Nelder-Mead MADWE
#####

function f(x, weight, reps)
    success, sum_stats = summary_fn(x, 17, reps)
    if success == true
        return weight(sum_stats)
    else
        return Inf
    end
    
end

weighted_mad = WeightedEuclidean(true_slob_log_returns, get_summary_stats, ones(17), "MAD")
sim_sum_stats = zeros(17, 100, n_replications)
Random.seed!(716527182)
for i in 1:size(sim_sum_stats, 2)
    params = rand(Int.(rand(MersenneTwister(), UInt32)), prior)
    success, sum_stats = summary_fn(params, 17, n_replications)
    sim_sum_stats[:,i,:] = sum_stats
end

inv_weights = [AdaptiveABC.MAD(sim_sum_stats[i, :, :]) for i in 1:17]  # For each statistic, calculate the MAD across all n_sim, n_rep values       
weights = 1.0 ./ inv_weights

weighted_mad.weights = weights
f_mad(x, grad) = f(x, weighted_mad, n_replications)
# D, σ, nu, μ, α
lower = [0.5, 0.1, 0.05, 0.1, 50.0]
upper = [3.0, 3.0, 0.95, 3.0, 200.0]

nm_reps = 10
results = zeros(size(lower, 1), nm_reps)
Random.seed!(627267162767)
for i in 1:nm_reps
    println("Iteration $i")
    opt = Opt(:LN_NELDERMEAD, size(lower, 1))
    opt.lower_bounds = lower
    opt.upper_bounds = upper
    opt.xtol_rel = 1e-6
    opt.min_objective = f_mad
    init_params = rand(Int.(rand(MersenneTwister(), UInt32)), prior)

    (minf, minx, ret) = NLopt.optimize(opt, init_params)

    results[:,i] = minx
end
println()
println("Nelder-Mead MADWE: ")
print(round.(mean(results, dims=2), digits=3))
results
println()
# Nelder-Mead MADWE: 
# [1.819; 1.711; 0.608; 1.786; 143.291]

