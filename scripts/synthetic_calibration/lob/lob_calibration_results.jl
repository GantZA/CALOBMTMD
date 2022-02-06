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
parameter_true = [D, σ, nu, μ]
parameter_names = ["D", "σ", "nu", "μ"]
plotting_parameter_names = ["D", "σ", "ν", "μ"]
n_replications = 10
Random.seed!(8929200)
# Constants
num_paths = 1; M = 400 ; T = 2299 ; p₀ = 238.75 ; L = 200
Δx = L / M ; λ = 1.0 ; Δt = (Δx^2) / (2.0 * D) ; α_lob = 0.0
println("Δx = $Δx and Δt = $Δt")



# Create model, simulate to obtain observation vector and then calculate summary statistics
true_lob_model = SLOB(num_paths,
    T, p₀, M, L, D, σ, nu, α_lob, SourceTerm(λ, μ))
true_lob_price_path = true_lob_model(7136)
true_lob_log_returns = diff(log.(true_lob_price_path[:,1]))
true_lob_summary_stats = get_summary_stats(true_lob_log_returns)

# plt = plot(
#     1:T + 1, true_lob_price_path[:,1], legend=false,
#     xlab="Time", ylab="Mid-Price",
# )

# Define Summary function used in AdaptiveABC. Takes in a vector of parameters and outputs a success boolean and the 
# simulated summart statistics
function summary_fn(parameters, n_summary_stats, n_replications)
    try
        D, σ, nu, μ = parameters
        model = SLOB(num_paths, T, p₀, M, L, D, σ, nu, α_lob, SourceTerm(λ, μ))
        summary_stats = Array{Float64,2}(undef, n_summary_stats, n_replications)
        for i in 1:n_replications
            sim_price_path = model()
            sim_obs = diff(log.(sim_price_path[:, 1]))
            summary_stats[:, i] = get_summary_stats(true_lob_log_returns, sim_obs)    
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
    Uniform(0.1, 3.0) # μ
])

#####
# Calibration Technique: ABC Rejection BBWM
#####

# Distance Function to measure distance between observed summary stats and simulated summary stats
boot_weight_matrix = BlockBootstrapWeightMatrix(122477, vcat([0], true_lob_log_returns), get_summary_stats, 100, 10_000)
weighted_bootstrap = WeightedBootstrap(vcat([0], true_lob_log_returns), get_summary_stats, boot_weight_matrix)



abc_input_wb = ABCInput(
    prior,
    parameter_names,
    true_lob_summary_stats,
    summary_fn,
    17,
    weighted_bootstrap
)

abc_reject_wb = JLD2.load_object("$(dirname(@__FILE__))/abc_reject_wb.jld2")

# abc_reject_wb = ABCRejection(
#     abc_input_wb,
#     3_000,
#     n_replications,
#     100.0,
#     parallel=true,
#     seed=2517413766247174462
# )

posterior_means = AdaptiveABC.parameter_means(abc_reject_wb)
println()
println("ABC rejection BBWM: D, σ, nu, μ")
print(round.(posterior_means[:,end], digits=3))
println()

# JLD2.save_object("$(dirname(@__FILE__))/abc_reject_wb.jld2", abc_reject_wb)

# ABC rejection BBWM: D, σ, nu, μ
# [1.551, 1.821, 0.503, 1.496]

param_inds = [1,2,3,4]
plt = plot(abc_reject_wb, iteration_colours=cgrad(:blues, 5, categorical=true),
    iterations=[1], params_true=parameter_true[param_inds], 
    prior_dists=abc_input_wb.prior.distribution[param_inds], 
    param_inds=param_inds, param_names=plotting_parameter_names[param_inds])

savefig(plt, "$(dirname(@__FILE__))/abc_reject_wb.pdf")
plt

#####
# Calibration Technique: ABC Rejection MADWE
#####
abc_input_we = ABCInput(
    prior,
    parameter_names,
    true_lob_summary_stats,
    summary_fn,
    17,
    WeightedEuclidean(true_lob_log_returns, get_summary_stats, ones(17), "MAD")
)

abc_reject_we = JLD2.load_object("$(dirname(@__FILE__))/abc_reject_we.jld2")
new_cutoff = 25.0

abc_reject_we.parameters = abc_reject_we.parameters[:, abc_reject_we.distances .<= new_cutoff]
abc_reject_we.n_successes = sum(abc_reject_we.distances .<= new_cutoff)
abc_reject_we.summary_stats = abc_reject_we.summary_stats[:, abc_reject_we.distances .<= new_cutoff, :]
abc_reject_we.weights = abc_reject_we.weights[abc_reject_we.distances .<= new_cutoff]

abc_reject_we.distances = abc_reject_we.distances[abc_reject_we.distances .<= new_cutoff]
# abc_reject_we = ABCRejection(
#     abc_input_we,
#     3_000,
#     n_replications,
#     23.0,
#     parallel=true,
#     seed=738173189
# )
posterior_means = AdaptiveABC.parameter_means(abc_reject_we)
println()
println("ABC rejection MADWE: D, σ, nu, μ")
print(round.(posterior_means[:,end], digits=3))
println()

# JLD2.save_object("$(dirname(@__FILE__))/abc_reject_we.jld2", abc_reject_we)

# ABC rejection MADWE: D, σ, nu, μ
# [1.747, 1.536, 0.51, 1.54]

param_inds = [1,2,3,4]
plt = plot(abc_reject_we, iteration_colours=cgrad(:blues, 5, categorical=true),
    iterations=[1], params_true=parameter_true[param_inds], 
    prior_dists=abc_input_we.prior.distribution[param_inds], 
    param_inds=param_inds, param_names=plotting_parameter_names[param_inds])

savefig(plt, "$(dirname(@__FILE__))/abc_reject_we.pdf")
plt


#####
# Calibration Technique: ABC-PMC BBWM
#####


# Distance Function to measure distance between observed summary stats and simulated summary stats
boot_weight_matrix = BlockBootstrapWeightMatrix(122477, vcat([0], true_lob_log_returns), get_summary_stats, 100, 10_000)
weighted_bootstrap = WeightedBootstrap(vcat([0], true_lob_log_returns), get_summary_stats, boot_weight_matrix)

abc_input_wb = ABCInput(
    prior,
    parameter_names,
    true_lob_summary_stats,
    summary_fn,
    17,
    weighted_bootstrap
)

abc_pmc_out_wb = JLD2.load_object("$(dirname(@__FILE__))/abc_pmc_out_wb.jld2")
# abc_pmc_out_wb = ABC_PMC(
#     abc_input_wb,
#     100,
#     n_replications,
#     0.25,
#     10_000,
#     10;
#     parallel=true,
#     batch_size=250,
#     seed=918731553
# )
posterior_means = AdaptiveABC.parameter_means(abc_pmc_out_wb)

println()
println("ABC-PMC BBWM: D, σ, nu, μ")
print(round.(posterior_means[:,end], digits=3))
println()

# JLD2.save_object("$(dirname(@__FILE__))/abc_pmc_out_wb.jld2", abc_pmc_out_wb)

# ABC-PMC BBWM: D, σ, nu, μ
# [1.086, 1.596, 0.457, 1.279]

param_inds = [1,2,3,4]
plt = plot(abc_pmc_out_wb, iteration_colours=cgrad(:blues, 5, categorical=true),
    iterations=[1, abc_pmc_out_wb.n_iterations ÷ 2, abc_pmc_out_wb.n_iterations], params_true=parameter_true[param_inds], 
    prior_dists=abc_input_wb.prior.distribution[param_inds], 
    param_inds=param_inds, param_names=plotting_parameter_names[param_inds])
plt
savefig(plt, "$(dirname(@__FILE__))/abc_pmc_wb.pdf")  

#####
# Calibration Technique: ABC-PMC MADWE
#####
abc_input_we = ABCInput(
    prior,
    parameter_names,
    true_lob_summary_stats,
    summary_fn,
    17,
    WeightedEuclidean(true_lob_log_returns, get_summary_stats, ones(17), "MAD")
)


abc_pmc_out_we = JLD2.load_object("$(dirname(@__FILE__))/abc_pmc_out_we.jld2")
# abc_pmc_out_we = ABC_PMC(
#     abc_input_we,
#     100,
#     n_replications,
#     0.15,
#     10_000,
#     10;
#     parallel=true,
#     batch_size=250,
#     seed=83912738712
# )
posterior_means = AdaptiveABC.parameter_means(abc_pmc_out_we)
println("")
println("ABC-PMC MADWE: D, σ, nu, μ")
print(round.(posterior_means[:,end], digits=3))
println("")
# ABC-PMC MADWE: D, σ, nu, μ
# [1.182, 1.767, 0.566, 1.75]

# JLD2.save_object("$(dirname(@__FILE__))/abc_pmc_out_we.jld2", abc_pmc_out_we)

param_inds = [1,2,3,4]
plt = plot(abc_pmc_out_we, iteration_colours=cgrad(:blues, 5, categorical=true),
    iterations=[1, abc_pmc_out_we.n_iterations ÷ 2, abc_pmc_out_we.n_iterations], params_true=parameter_true[param_inds], 
    prior_dists=abc_input_we.prior.distribution[param_inds], 
    param_inds=param_inds, param_names=plotting_parameter_names[param_inds])
savefig(plt, "$(dirname(@__FILE__))/abc_pmc_we.pdf")  
plt

#####
# Calibration Technique: Nelder-Mead BBWM
#####

# Distance Function to measure distance between observed summary stats and simulated summary stats
boot_weight_matrix = BlockBootstrapWeightMatrix(122477, vcat([0], true_lob_log_returns), get_summary_stats, 100, 10_000)
weighted_bootstrap = WeightedBootstrap(vcat([0], true_lob_log_returns), get_summary_stats, boot_weight_matrix)

function f(x, weight, reps)
    success, sum_stats = summary_fn(x, 17, reps)
    return weight(sum_stats)
end

f_wb(x, grad) = f(x, weighted_bootstrap, n_replications)
# D, σ, nu, μ
lower = [0.5, 0.1, 0.05, 0.1]
upper = [3.0, 3.0, 0.95, 3.0]

nm_reps = 10
results = zeros(4, nm_reps)
Random.seed!(736161311)
for i in 1:nm_reps
    println("Iteration $i")
    opt = Opt(:LN_NELDERMEAD, 4)
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
# [1.778; 1.944; 0.719; 1.756]

#####
# Calibration Technique: Nelder-Mead MADWE
#####

weighted_mad = WeightedEuclidean(true_lob_log_returns, get_summary_stats, ones(17), "MAD")
sim_sum_stats = zeros(17, 100, n_replications)
Random.seed!(1773616333)
for i in 1:size(sim_sum_stats, 2)
    params = rand(Int.(rand(MersenneTwister(), UInt32)), prior)
    success, sum_stats = summary_fn(params, 17, n_replications)
    sim_sum_stats[:,i,:] = sum_stats
end

inv_weights = [AdaptiveABC.MAD(sim_sum_stats[i, :, :]) for i in 1:17]  # For each statistic, calculate the MAD across all n_sim, n_rep values       
weights = 1.0 ./ inv_weights

weighted_mad.weights = weights
f_mad(x, grad) = f(x, weighted_mad, n_replications)
# D, σ, nu, μ
lower = [0.5, 0.1, 0.05, 0.1]
upper = [3.0, 3.0, 0.95, 3.0]

nm_reps = 10
results = zeros(4, nm_reps)
Random.seed!(9918133091)
for i in 1:nm_reps
    println("Iteration $i")
    opt = Opt(:LN_NELDERMEAD, 4)
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
# [2.373; 1.924; 0.52; 1.295]
