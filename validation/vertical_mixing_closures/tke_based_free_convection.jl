using Plots
using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization, TKEBasedVerticalDiffusivity
using Oceananigans.TurbulenceClosures: RiDependentDiffusivityScaling

function run_free_convection(; Nz=32, stop_time=2days, Qᵇ=1e-8, Lz=128, N²=1e-5, Δt=10)

    grid = RegularRectilinearGrid(size=Nz, z=(-Lz, 0), topology=(Flat, Flat, Bounded))

    closure = TKEBasedVerticalDiffusivity()
    b_bcs = TracerBoundaryConditions(grid; top = FluxBoundaryCondition(Qᵇ))

    model = HydrostaticFreeSurfaceModel(grid = grid,
                                        tracers = (:b, :e),
                                        buoyancy = BuoyancyTracer(),
                                        boundary_conditions = (b=b_bcs,),
                                        closure = closure)
    
    set!(model, b = (x, y, z) -> N² * z)
                      
    simulation = Simulation(model, Δt = Δt, stop_time = stop_time)

    run!(simulation)

    return simulation
end

plot_free_convection(simulation; kwargs...) = plot_free_convection!((b=plot(), e=plot(), K=plot()), simulation; kwargs...)

function plot_free_convection!(plots, simulation; label=@sprintf("Nz=%d", simulation.model.grid.Nz), color=:blue)

    model = simulation.model

    z = znodes(model.tracers.b)
    b = view(interior(model.tracers.b), 1, 1, :)
    e = view(interior(model.tracers.e), 1, 1, :)
    Kc = view(interior(model.diffusivities.Kᶜ), 1, 1, :)
    Ke = view(interior(model.diffusivities.Kᵉ), 1, 1, :)

    common_kwargs = (; color = color, legend = :bottomright, ylabel = "z (m)")

    plot!(plots.b, b, z,  linewidth = 2, color = color, label = @sprintf("%s t = %s", label, prettytime(model.clock.time)))
    plot!(plots.e, e, z,  linewidth = 2, color = color, label = @sprintf("%s t = %s", label, prettytime(model.clock.time)))
    plot!(plots.K, Kc, z, linewidth = 2, color = color, label = @sprintf("%s Kᶜ, t = %s", label, prettytime(model.clock.time)), linestyle=:dash)
    plot!(plots.K, Ke, z, linewidth = 3, color = color, label = @sprintf("%s Kᵉ, t = %s", label, prettytime(model.clock.time)), alpha=0.6)

    return plots
end

plots = plot_free_convection(run_free_convection(Nz=8), label="Nz=8", color=:blue)

@time plot_free_convection!(plots, run_free_convection(Nz=16),  plots, color=:green)
@time plot_free_convection!(plots, run_free_convection(Nz=32),  plots, color=:cyan)
@time plot_free_convection!(plots, run_free_convection(Nz=64),  plots, color=:red)
@time plot_free_convection!(plots, run_free_convection(Nz=128), plots, color=:black)

summary = plot(b_plot, e_plot, K_plot, layout=(1, 3), size=(1200, 600))

display(summary)
