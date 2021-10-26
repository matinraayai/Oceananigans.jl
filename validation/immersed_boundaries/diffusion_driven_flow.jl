using Printf
using Oceananigans
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, GridFittedBoundary, mask_immersed_field!
using Oceananigans: fields

Nz = 32   # Resolution
Bu = 10  # Slope Burger number
f = 1     # Coriolis parameter
κ = 0.1   # Diffusivity and viscosity (Prandtl = 1)

sin²θ = 0.5 # Slope of 45 deg

# Buoyancy frequency
@show N² = Bu * f^2 / sin²θ

# Time-scale for diffusion over 1 unit
τ = 1 / κ

underlying_grid = RegularRectilinearGrid(size = (2Nz, Nz),
                                         x = (-3, 3),
                                         z = (-1, 2),
                                         topology = (Bounded, Flat, Bounded))

@inline slope(x, y) = min(x, 1)
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(slope))

model = NonhydrostaticModel(architecture = CPU(),
                            grid = grid,
                            advection = CenteredSecondOrder(),
                            closure = IsotropicDiffusivity(ν=κ, κ=κ),
                            tracers = :b,
                            coriolis = FPlane(f=1),
                            buoyancy = BuoyancyTracer())

# Linear stratification
set!(model, b = (x, y, z) -> N² * z)

start_time = [time_ns()]

progress(s) = @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|w|: %.2e",
                             100 * s.model.clock.time / s.stop_time, s.model.clock.iteration,
                             s.model.clock.time, maximum(abs, model.velocities.w))

Δt = 0.1 * grid.Δx^2 / model.closure.κ.b

simulation = Simulation(model, Δt = Δt, stop_time = 1τ, progress = progress, iteration_interval = 100)

simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                      schedule = TimeInterval(0.1τ),
                                                      prefix = "diffusion_driven_flow",
                                                      force = true)
                        
run!(simulation)

@info """
    Simulation complete.
    Runtime: $(prettytime(simulation.run_time))
    Output: $(abspath(simulation.output_writers[:fields].filepath))
"""

u, v, w, b = fields(model)

for f in fields(model)
    mask_immersed_field!(f, NaN)
end

u = Array(interior(u))[:, 1, :]
v = Array(interior(v))[:, 1, :]
w = Array(interior(w))[:, 1, :]
b = Array(interior(b))[:, 1, :]

fluid_u = filter(isfinite, u[:])
fluid_v = filter(isfinite, v[:])
fluid_w = filter(isfinite, w[:])
fluid_b = filter(isfinite, b[:])

max_u = maximum(abs, fluid_u)
max_v = maximum(abs, fluid_v)
max_w = maximum(abs, fluid_w)
max_b = maximum(fluid_b)
min_b = minimum(fluid_b)

using GLMakie

fig = Figure(resolution=(1800, 1200))

ax_b = Axis(fig[1, 1], title="Buoyancy")
hm_b = heatmap!(ax_b, b, colorrange=(min_b, max_b), colormap=:thermal)
cb_b = Colorbar(fig[1, 2], hm_b)

ax_u = Axis(fig[1, 3], title="x-velocity")
hm_u = heatmap!(ax_u, u, colorrange=(-max_u, max_u), colormap=:balance)
cb_u = Colorbar(fig[1, 4], hm_u)

ax_v = Axis(fig[2, 1], title="y-velocity")
hm_v = heatmap!(ax_v, v, colorrange=(-max_v, max_v), colormap=:balance)
cb_v = Colorbar(fig[2, 2], hm_v)

ax_w = Axis(fig[2, 3], title="z-velocity")
hm_w = heatmap!(ax_w, w, colorrange=(-max_w, max_w), colormap=:balance)
cb_w = Colorbar(fig[2, 4], hm_w)

ax_t = fig[0, :] = Label(fig, "Diffusion driven flow at t = $(model.clock.time)")

display(fig)

