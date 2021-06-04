using Printf, JLD2, Plots
using Oceananigans
using Oceananigans.Units: hours, day, days, minutes, kilometers
using Oceananigans.TurbulenceClosures: TKEBasedVerticalDiffusivity, HorizontallyCurvilinearAnisotropicBiharmonicDiffusivity, ConvectiveAdjustmentParameters

# ## The grid

Lz = 400
Nz = 32
Nh = 48
ς(k) = ((k - 1) / Nz)^1
z_faces(k) = - Lz * (ς(k) - 1)

grid = RegularRectilinearGrid(size = (1, 1, Nz),
                              halo = (3, 3, 3),
                              x = (0, 100kilometers),
                              y = (-50kilometers, 50kilometers),
                              z = (-Lz, 0),
                              topology = (Periodic, Bounded, Bounded))

#=
grid = VerticallyStretchedRectilinearGrid(size = (Nh, Nh, Nz),
                                          halo = (3, 3, 3),
                                          x = (0, 100kilometers),
                                          y = (-50kilometers, 50kilometers),
                                          z_faces = collect(grid.zF[1:Nz+1]),
                                          topology = (Periodic, Bounded, Bounded))
=#

#coriolis = BetaPlane(latitude = -45)
coriolis = FPlane(latitude = -45)

#####
##### Initial condition
#####

# Baroclinically unstable base state

α = coriolis.f
ℓ = grid.Ly / 10
f = coriolis.f
N² = 1e-5
ϵ = 0.1

# Geostrophic initial condition
# ψ = - ϵ * f * ℓ² * tanh(y / ℓ) * (z / Lz + 1)
#
# b = f ∂z ψ = - ϵ * f * ℓ² / Lz * tanh(y / ℓ)
# u = - ∂y ψ = ϵ * f * ℓ² * sech(y / ℓ)^2 * (z / Lz + 1)

## Background fields are defined via functions of x, y, z, t, and optional parameters
uᵢ(x, y, z) = - ϵ * f * ℓ * sech(y / ℓ)^2 * (z / Lz + 1) * (1 + 1e-2 * rand())
bᵢ(x, y, z) = + ϵ * (f * ℓ)^2 / Lz * tanh(y / ℓ) + N² * z

######
###### Boundary conditions
######

#=
surface_buoyancy_flux(x, y, t, p) = p.Qᵇ #* tanh(y / p.δ)
surface_bc_b = FluxBoundaryCondition(surface_buoyancy_flux, parameters = (Qᵇ = 1e-8, δ = grid.Ly / 20))

b_north_south(x, z, t, y_boundary) = bᵢ(x, y_boundary, z)
northern_bc_b = FluxBoundaryCondition(b_north_south, parameters = + grid.Ly / 2)
southern_bc_b = FluxBoundaryCondition(b_north_south, parameters = - grid.Ly / 2)
b_bcs = TracerBoundaryConditions(grid, top = surface_bc_b) #, north = northern_bc_b, south = southern_bc_b)
=#

#=
cᴰ = 1e-4 # quadratic drag coefficient
@inline drag_u(x, y, t, u, v, cᴰ) = - cᴰ * u * sqrt(u^2 + v^2)
@inline drag_v(x, y, t, u, v, cᴰ) = - cᴰ * v * sqrt(u^2 + v^2)

drag_bc_u = FluxBoundaryCondition(drag_u, field_dependencies=(:u, :v), parameters=cᴰ)
drag_bc_v = FluxBoundaryCondition(drag_v, field_dependencies=(:u, :v), parameters=cᴰ)

u_bcs = UVelocityBoundaryConditions(grid, bottom = drag_bc_u)
v_bcs = VVelocityBoundaryConditions(grid, bottom = drag_bc_v)
=#

b_bcs = TracerBoundaryConditions(grid, top = FluxBoundaryCondition(1e-8))

# ## Turbulence closures

κ₄h = 1e-2 / day * grid.Δx^4 # [m⁴ s⁻¹] horizontal hyperviscosity and hyperdiffusivity
κ₂h = 1e-4 / day * grid.Δx^2 # [m² s⁻¹] horizontal Laplacian diffusivity

convective_adjustment = ConvectiveAdjustmentParameters(Cᴬc = 100.0, Cᴬu = 10.0, Cᴬe = 10.0)
vertical_diffusivity = TKEBasedVerticalDiffusivity(convective_adjustment=convective_adjustment)
#horizontal_diffusivity = AnisotropicBiharmonicDiffusivity(νh=κ₄h, κh=κ₄h)

#horizontal_diffusivity = nothing
horizontal_diffusivity = AnisotropicDiffusivity(νh=κ₂h, κh=κ₂h, κz=1e-2)

# ## Model instantiation
#
# We instantiate the model with the fifth-order WENO advection scheme, a 3rd order
# Runge-Kutta time-stepping scheme, and a `BuoyancyTracer`.

#model = IncompressibleModel(
#              advection = UpwindBiasedFifthOrder(),

model = HydrostaticFreeSurfaceModel(
           architecture = CPU(),
                   grid = grid,
           #free_surface = ExplicitFreeSurface(gravitational_acceleration=0.1),
           free_surface = ImplicitFreeSurface(gravitational_acceleration=9.8),
     momentum_advection = UpwindBiasedFifthOrder(),
       tracer_advection = UpwindBiasedFifthOrder(),
               coriolis = coriolis,
                tracers = (:b, :e),
               buoyancy = BuoyancyTracer(),
                closure = vertical_diffusivity, # horizontal_diffusivity),
    boundary_conditions = (b=b_bcs,)
)

set!(model, u = uᵢ, b = bᵢ)

Δt = 0.1minutes

wizard = TimeStepWizard(cfl=0.2, Δt=Δt, max_change=1.1, max_Δt=Δt)

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

simulation = Simulation(model,
                        Δt = wizard,
                        iteration_interval = 10,
                        #stop_time = 16days,
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
                                                      #schedule = TimeInterval(0.2days),
                                                      schedule = IterationInterval(10),
                                                        prefix = "eddying_channel",
                                                         force = true)


run!(simulation)

u, v, w = model.velocities

xζ, yζ, zζ = nodes(ζ)
xu, yu, zu = nodes(u)
xδ, yδ, zδ = nodes(δ)

file = jldopen("eddying_channel.jld2")

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

    @info "Loading frame $i of $(length(iterations))..."

    ## Load 3D fields from file
    t = file["timeseries/t/$iter"]
    R_snapshot = file["timeseries/ζ/$iter"] ./ coriolis.f
    δ_snapshot = file["timeseries/δ/$iter"]
    b_snapshot = file["timeseries/b/$iter"]
    e_snapshot = file["timeseries/e/$iter"]
    u_snapshot = file["timeseries/u/$iter"]

    surface_R = R_snapshot[:, :, grid.Nz]
    surface_δ = δ_snapshot[:, :, grid.Nz]
    surface_u = u_snapshot[:, :, grid.Nz]

    slice_R = R_snapshot[1, :, :]
    slice_δ = δ_snapshot[1, :, :]
    slice_u = u_snapshot[1, :, :]
    slice_b = b_snapshot[1, :, :]

    Rlim = 0.5 * maximum(abs, R_snapshot) + 1e-9
    δlim = 0.5 * maximum(abs, δ_snapshot) + 1e-9
    ulim = 0.5 * maximum(abs, u_snapshot) + 1e-9
    blim = 0.5 * maximum(abs, slice_b) + 1e-9

    Rlevels = nice_divergent_levels(R_snapshot, Rlim)
    δlevels = nice_divergent_levels(δ_snapshot, δlim)
    ulevels = nice_divergent_levels(u_snapshot, ulim)
    blevels = nice_divergent_levels(slice_b, blim)

    @info @sprintf("Drawing frame %d from iteration %d: max(ζ̃ / f) = %.3f \n",
                   i, iter, maximum(abs, surface_R))

    xy_kwargs = (xlims = (0, grid.Lx), ylims = (-grid.Ly/2, grid.Ly/2),
                 xlabel = "x (m)", ylabel = "y (m)",
                 aspectratio = 1,
                 linewidth = 0, color = :balance, legend = false)

    xz_kwargs = (xlims = (-grid.Ly/2, grid.Ly/2), ylims = (-grid.Lz, 0),
                 xlabel = "y (m)", ylabel = "z (m)",
                 aspectratio = grid.Ly / grid.Lz * 0.5,
                 linewidth = 0, color = :balance, legend = false)

    #δ_xy = contourf(xδ, yδ, surface_δ'; clims=(-δlim, δlim), levels=δlevels, xy_kwargs...)
    #δ_xz = contourf(xδ, zδ, slice_δ'; clims=(-δlim, δlim), levels=δlevels, xz_kwargs...)
    
    # R_xy = contourf(xζ, yζ, surface_R'; clims=(-Rlim, Rlim), levels=Rlevels, xy_kwargs...)
    # u_xy = contourf(xu, yu, surface_u'; clims=(-ulim, ulim), levels=ulevels, xy_kwargs...)

    # R_xz = contourf(yζ, zζ, slice_R'; clims=(-Rlim, Rlim), levels=Rlevels, xz_kwargs...)
    # u_xz = contourf(yu, zu, slice_u'; clims=(-ulim, ulim), levels=ulevels, xz_kwargs...)
    # b_xz = contourf(yδ, zδ, slice_b'; clims=(-blim, blim), levels=blevels, xz_kwargs...)

    b_z = plot(b_snapshot[1, 1, :], zδ, legend=nothing) #; clims=(-blim, blim), levels=blevels, xz_kwargs...)
    e_z = plot(e_snapshot[1, 1, :], zδ, legend=nothing) #; clims=(-blim, blim), levels=blevels, xz_kwargs...)
    u_z = plot(u_snapshot[1, 1, :], zu, legend=nothing) #; clims=(-blim, blim), levels=blevels, xz_kwargs...)

    plot(b_z, e_z, u_z, layout = (1, 3))

    #=
    plot(R_xy, u_xy, u_xz, b_xz,
           size = (1200, 1000),
           link = :x,
         layout = Plots.grid(2, 2, heights=[0.5, 0.5, 0.2, 0.2]),
          title = [@sprintf("ζ(t=%s) / f", prettytime(t)) @sprintf("u(t=%s) (s⁻¹)", prettytime(t)) "" "Buoyancy"])
    =#

    iter == iterations[end] && close(file)
end

mp4(anim, "eddying_channel.mp4", fps = 8) # hide

