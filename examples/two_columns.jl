using Printf, JLD2, Plots
using Oceananigans
using Oceananigans.Units: hours, day, days, minutes, kilometers
using Oceananigans.TurbulenceClosures: TKEBasedVerticalDiffusivity, VerticallyImplicitTimeDiscretization

Lz = 400
Nz = 32

grid = RegularRectilinearGrid(size = (1, Nz),
                              x = (0, 100kilometers),
                              y = (-50kilometers, 50kilometers),
                              z = (-Lz, 0),
                              topology = (Flat, Periodic, Bounded))

#=
grid = VerticallyStretchedRectilinearGrid(size = (1, Nz),
                                          x = (0, 100kilometers),
                                          y = (-50kilometers, 50kilometers),
                                          z_faces = collect(grid.zF[1:Nz+1]),
                                          topology = (Flat, Periodic, Bounded))
                                          =#

coriolis = FPlane(latitude = -45)
N² = 1e-5
bᵢ(x, y, z) = N² * z

b_bcs = TracerBoundaryConditions(grid, top = FluxBoundaryCondition(1e-8))

closure = (#TKEBasedVerticalDiffusivity(),
           AnisotropicDiffusivity(κz = 1e-2, time_discretization=VerticallyImplicitTimeDiscretization()))

model = HydrostaticFreeSurfaceModel(
           architecture = CPU(),
                   grid = grid,
               coriolis = coriolis,
                tracers = (:b, :e),
               buoyancy = BuoyancyTracer(),
                closure = closure,
    boundary_conditions = (b=b_bcs,)
)

set!(model, b = bᵢ, u = (x, y, z) -> 1 + z/Lz)
#set!(model, b = bᵢ)

Δt = 5minutes

wizard = TimeStepWizard(cfl=0.1, Δt=Δt, max_change=1.1, max_Δt=Δt)

CFL = AdvectiveCFL(wizard)

wall_clock = [time_ns()]

function progress(sim)
    @printf("i: % 6d, sim time: % 10s, wall time: % 10s, Δt: % 10s, CFL: %.2e\n",
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[1])),
            prettytime(sim.Δt.Δt),
            CFL(sim.model))

    wall_clock[1] = time_ns()

    return nothing
end

simulation = Simulation(model, Δt = wizard, iteration_interval = 10,
                                                stop_iteration = 10000,
                                                      progress = progress)

# ### Output
#
# To visualize the baroclinic turbulence ensuing in the Eady problem,
# we use `ComputedField`s to diagnose and output vertical vorticity and divergence.
# Note that `ComputedField`s take "AbstractOperations" on `Field`s as input:

u, v, w = model.velocities # unpack velocity `Field`s

## Vertical vorticity [s⁻¹]
ζ = ComputedField(∂x(v) - ∂y(u))

## Horizontal divergence, or ∂x(u) + ∂y(v) [s⁻¹]
δ = ComputedField(-∂z(w))

outputs = merge(model.velocities, model.tracers, (ζ=ζ, δ=δ))

simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                      schedule = IterationInterval(floor(Int, simulation.stop_iteration / 100)),
                                                      #schedule = TimeInterval(0.2days),
                                                        prefix = "two_columns",
                                                         force = true)

run!(simulation)

u, v, w = model.velocities

xu, yu, zu = nodes(u)

file = jldopen("two_columns.jld2")

iterations = parse.(Int, keys(file["timeseries/t"]))

function nice_divergent_levels(c, clim, nlevels=31)
    !isfinite(clim) && (clim = 1)
    levels = range(-clim, stop=clim, length=nlevels)
    cmax = maximum(abs, c)
    clim < cmax && (levels = vcat([-cmax], levels, [cmax]))
    return levels
end

@info "Making an animation from saved data..."

anim = @animate for (i, iter) in enumerate(iterations)

    ## Load 3D fields from file
    t = file["timeseries/t/$iter"]
    b_snapshot = file["timeseries/b/$iter"]
    u_snapshot = file["timeseries/u/$iter"]

    @info @sprintf("Drawing frame %d from iteration %d\n", i, iter)

    bz = plot(b_snapshot[1, 1, :], zu, legend=nothing)
    uz = plot(u_snapshot[1, 1, :], zu, legend=nothing)

    plot(bz, uz, layout = (2, 1))

    iter == iterations[end] && close(file)
end

mp4(anim, "two_columns.mp4", fps = 8) # hide

