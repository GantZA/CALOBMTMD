using SequentialLOB
using Plots
pyplot()
using PyPlot
pygui(true)

num_paths = 1; M = 800 ; T = 2299 ; p₀ = 238.75 ; L = 200 ; Δx = L / M ; λ = 1.0
D = 0.5 ; Δx = L / M ; Δt = (Δx^2) / (2.0 * D) ; σ = 1.5 ; μ = 0.03 ; nu = 0.5
α = 0.0 ; α_slob = 500.0

slob_model = SLOB(num_paths, T, p₀, M, L, D, σ, nu, α, SourceTerm(λ, μ)) 
    
lob_densities, raw_price_paths, sample_price_paths, P⁺s, P⁻s, Ps = slob_model(true, 5487)

num_xs = Int(M / 8 + 1)
time_steps_range = 0:3000:30000

x_s = repeat(collect(range(p₀ - L / 2, stop=p₀ + L / 2, length=num_xs)), inner=11)
y_s = repeat(collect(time_steps_range), outer=num_xs)
z_s = vec(transpose(lob_densities[Int.(range(1, M + 1, length=num_xs)), (time_steps_range) .+ 1, 1]))

plt = Plots.plot(
    x_s, y_s, z_s, st=:surface,
    size=(1000, 800),
        xlabel="Price" , ylabel="Time",
        zlabel="Density (φ)", camera=(45, 15),
        xguidefontsize=14,yguidefontsize=14,zguidefontsize=14,
        xtickfontsize=8,ytickfontsize=8,ztickfontsize=8,
        dpi=200, zguidefontrotation=90, formatter=:plain, 
        c=:redsblues, legend=:none
)

Plots.savefig(plt, "$(dirname(@__FILE__))/lob_density_values_sample.pdf")



slob_model = SLOB(num_paths, T, p₀, M, L, D, σ, nu, α_slob, SourceTerm(λ, μ)) 
    
lob_densities, raw_price_paths, sample_price_paths, P⁺s, P⁻s, Ps = slob_model(true, 173781)

num_xs = Int(M / 8 + 1)
time_steps_range = 0:3000:30000

x_s = repeat(collect(range(p₀ - L / 2, stop=p₀ + L / 2, length=num_xs)), inner=11)
y_s = repeat(collect(time_steps_range), outer=num_xs)
z_s = vec(transpose(lob_densities[Int.(range(1, M + 1, length=num_xs)), (time_steps_range) .+ 1, 1]))

plt = Plots.plot(
    x_s, y_s, z_s, st=:surface,
    size=(1000, 800),
        xlabel="Price" , ylabel="Time",
        zlabel="Density (φ)", camera=(35, 15),
        xguidefontsize=14,yguidefontsize=14,zguidefontsize=14,
        xtickfontsize=8,ytickfontsize=8,ztickfontsize=8,
        dpi=200, zguidefontrotation=90, formatter=:plain, 
        c=:redsblues, legend=:none
)

Plots.savefig(plt, "$(dirname(@__FILE__))/slob_density_values_sample.pdf")

anim = @animate for i = 1:40
    plt1 = Plots.plot(
        x_s, y_s, z_s, st=:surface,
        size=(1000, 800), title="Camera Angle=($(i * 5), 15)",
            xlabel="Price" , ylabel="Time",
            zlabel="Density (φ)", camera=(i * 5, 15),
            xguidefontsize=14,yguidefontsize=14,zguidefontsize=14,
            xtickfontsize=8,ytickfontsize=8,ztickfontsize=8,
            dpi=100, zguidefontrotation=90, formatter=:plain, 
            c=:redsblues, legend=:none
    )
end
gif(anim, "$(dirname(@__FILE__))/slob_density_values.gif", fps=8)