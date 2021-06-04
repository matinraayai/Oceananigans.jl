using Plots
using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization, TKEBasedVerticalDiffusivity
using Oceananigans.TurbulenceClosures: RiDependentDiffusivityScaling

function run_free_convection(; Nz=32, stop_time=48hours, Qᵇ = 1e-8)

    grid = RegularRectilinearGrid(size=Nz, z=(-128, 0), topology=(Flat, Flat, Bounded))

    closure = TKEBasedVerticalDiffusivity()
    b_bcs = TracerBoundaryConditions(grid; top = FluxBoundaryCondition(Qᵇ))

    model = HydrostaticFreeSurfaceModel(grid = grid,
                                        tracers = (:b, :e),
                                        buoyancy = BuoyancyTracer(),
                                        boundary_conditions = (b=b_bcs,),
                                        closure = closure)
                                        
    N² = 1e-5
    bᵢ(x, y, z) = N² * z
    set!(model, b = bᵢ)
                      
    simulation = Simulation(model, Δt = 10.0, stop_time = 48hours)

    run!(simulation)

    return simulation
end

function plot_free_convection!(simulation, plots=nothing; label="", color=:blue)

    model = simulation.model

    z = znodes(model.tracers.b)
    b = view(interior(model.tracers.b), 1, 1, :)
    e = view(interior(model.tracers.e), 1, 1, :)
    Kc = view(interior(model.diffusivities.Kᶜ), 1, 1, :)
    Ke = view(interior(model.diffusivities.Kᵉ), 1, 1, :)

    common_kwargs = (color = color, legend = :bottomright, ylabel = "z (m)")


    if plots === nothing
        b_plot = plot(b,  z, linewidth = 2, label = @sprintf("%s t = %s",     label, prettytime(model.clock.time)), xlabel="Buoyancy", common_kwargs...)
        e_plot = plot(e,  z, linewidth = 2, label = @sprintf("%s t = %s",     label, prettytime(model.clock.time)), xlabel="TKE", common_kwargs...)
        K_plot = plot(Kc, z, linewidth = 2, label = @sprintf("%s Kᶜ, t = %s", label, prettytime(model.clock.time)), xlabel="Diffusivities", linestyle=:dash, common_kwargs...)
        plot!(K_plot, Ke, z, linewidth = 3, label = @sprintf("%s Kᵉ, t = %s", label, prettytime(model.clock.time)), alpha=0.6, common_kwargs...)

        return b_plot, e_plot, K_plot
    else
        plot!(plots.b, b, z,  linewidth = 2, color = color, label = @sprintf("%s t = %s", label, prettytime(model.clock.time)))
        plot!(plots.e, e, z,  linewidth = 2, color = color, label = @sprintf("%s t = %s", label, prettytime(model.clock.time)))
        plot!(plots.K, Kc, z, linewidth = 2, color = color, label = @sprintf("%s Kᶜ, t = %s", label, prettytime(model.clock.time)), linestyle=:dash)
        plot!(plots.K, Ke, z, linewidth = 3, color = color, label = @sprintf("%s Kᵉ, t = %s", label, prettytime(model.clock.time)), alpha=0.6)

        return nothing
    end
end

b_plot, e_plot, K_plot = plot_free_convection!(run_free_convection(Nz=8), label="Nz=8", color=:blue)

plots = (b=b_plot, e=e_plot, K=K_plot)

plot_free_convection!(run_free_convection(Nz=8),  plots, label="Nz=8",  color=:green)
plot_free_convection!(run_free_convection(Nz=32),  plots, label="Nz=32",  color=:cyan)
plot_free_convection!(run_free_convection(Nz=64),  plots, label="Nz=64",  color=:red)
plot_free_convection!(run_free_convection(Nz=128), plots, label="Nz=128", color=:black)

summary = plot(b_plot, e_plot, K_plot, layout=(1, 3), size=(1200, 600))

display(summary)
