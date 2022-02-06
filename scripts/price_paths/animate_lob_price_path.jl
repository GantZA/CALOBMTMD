using SequentialLOB
using StylizedFacts
using Plots

# Configuration Arguments
num_paths = 1
M = 400
T = 2299
p₀ = 238.75
L = 200
Δx = L / M 

# Free-Parameters
λ = 1.0
D = 1.0
Δt = (Δx^2) / (2.0 * D)
σ = 1.5
μ = 1.0
nu = 0.5
α_slob = 100.0
α_lob = 0.0
println(Δx)
println(Δt)

lob_model = SLOB(num_paths,
T, p₀, M, L, D, σ, nu, α_lob, SourceTerm(λ, μ))

lob_densities, raw_price_paths, sample_price_paths, P⁺s, P⁻s, Ps = lob_model(true, 7136) 

anim = @animate for i = 1:100:15000
    l = @layout [a ; b]
    time_step = round(i * lob_model.Δt, digits=0)
    time_step_ind = floor(Int, time_step)
    plt1 = plot(1:time_step_ind, sample_price_paths[1:time_step_ind,1],
        legend=false, ylab="Price", xlab="Time") ;
    plt2 = plot(lob_model.x, lob_densities[:,i,1], legend=false, title="t=$time_step",
    xlab="Price", ylab="LOB Density") ;
    plot!(lob_model.x, x -> 0) ;
    plot(plt1, plt2, layout=l)
end
gif(anim, "$(dirname(@__FILE__))/LOB.gif", fps=8)