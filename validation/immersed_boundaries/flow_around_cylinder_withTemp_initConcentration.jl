# # Steady-state flow around a cylinder in 2D using immersed boundaries

using Statistics
using Plots
using JLD2
using Printf

using Oceananigans
using Oceananigans.Advection
using Oceananigans.Grids
using Oceananigans.Fields
using Oceananigans.AbstractOperations
using Oceananigans.OutputWriters
using Oceananigans.BoundaryConditions

# setting boundary condition topology
topology=(Periodic, Periodic, Bounded)

# setting up 2D grid
grid = RegularRectilinearGrid(topology=topology, size=(500, 750, 1), x=(20, 40), y=(10, 40), z=(0, 1), halo=(3, 3, 3))


# reynolds number
Re = 40

#cylinder with center at (30,20)
const R = 1 # radius
vCent = [30.0, 20.0, 0.];
dist_cylinder(v) = sqrt((vCent[1]-v[1])^2+(vCent[2]-v[2])^2)-R # immersed solid
inside_cylinder(x, y, z) = ((x-vCent[1])^2 + (y-vCent[2])^2) <= R # immersed solid

# masking function

@inline heaviside(X) = ifelse(X < 0, zero(X), one(X))
@inline mask2nd(X) = heaviside(X) * X^2
function top_mask(x, y, z)
    y₁ = 40; y₀ = 30
    return mask2nd((y - y₀)/(y₁ - y₀))
end

full_sponge_1 = Relaxation(rate= 0.7, mask=top_mask, target=1)
full_sponge_0 = Relaxation(rate= 0.7, mask=top_mask, target=0)


# boundary conditions: inflow and outflow in y
v_bcs = VVelocityBoundaryConditions(grid,
                                    north = BoundaryCondition(NormalFlow,1.0),
                                    south = BoundaryCondition(NormalFlow,1.0))

T_bcs = TracerBoundaryConditions(grid, east = FluxBoundaryCondition(nothing), west = FluxBoundaryCondition(nothing))

# setting up incompressible model with immersed boundary
model = IncompressibleModel(timestepper = :RungeKutta3, 
                              advection = CenteredSecondOrder(),
                                   grid = grid,
                               buoyancy = nothing,
                                tracers = :T,
                                closure = IsotropicDiffusivity(ν=2/Re, κ=2/Re),
                    boundary_conditions = (v=v_bcs, T = T_bcs),
                                forcing = (u = full_sponge_0, v = full_sponge_1,),
                      immersed_boundary = dist_cylinder
                           )

# initial condition
# setting velocitiy to zero inside the cylinder and 1 everywhere else
v₀(x, y, z) = ifelse(inside_cylinder(x,y,z), 0., 1.)
T₀(x, y, z) = ifelse(inside_cylinder(x,y,z), 0., 1.)
set!(model, v=v₀, T=T₀)


progress(sim) = @info @sprintf("Iteration: % 4d, time: %.2f, max(v): %.5f, min(v): %.5f",
sim.model.clock.iteration,
sim.model.clock.time,
maximum(sim.model.velocities.v.data),
minimum(sim.model.velocities.v.data))

simulation = Simulation(model, Δt=5.7e-3, stop_time=20, iteration_interval=100, progress=progress)

# ## Output

simulation.output_writers[:fields] = JLD2OutputWriter(model,
                                                      merge(model.velocities, model.tracers),
                                                      schedule = TimeInterval(0.5),
                                                      prefix = "flow_around_cylinder_tempIBM_newBCs_51",
                                                      force = true)

# run it
run!(simulation)

# Analyze Results
file = jldopen(simulation.output_writers[:fields].filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))

u, v, w = model.velocities

xv, yv, zv = nodes(v)

# defining a function to mark boundary
function circle_shape(h, k, r)
            θ = LinRange(0,2*π,500)
            h.+r*sin.(θ),k.+r*cos.(θ)
end

@info "Making a neat movie of velocity..."

anim = @animate for (i, iteration) in enumerate(iterations)

    @info "Plotting frame $i from iteration $iteration..."
    
    t = file["timeseries/t/$iteration"]

    v_slice = file["timeseries/v/$iteration"][:, :, 1]
    @info maximum(v_slice) minimum(v_slice)
    v_max = maximum(abs, v_slice)
    v_lim = 0.8 * v_max

    v_levels = vcat([-v_max], range(-v_lim, stop=v_lim, length=50), [v_max])

    v_plot = contourf(xv, yv, v_slice';
                      linewidth = 0,
                          color = :balance,
                    aspectratio = 1,
                          title = @sprintf("v(x, y, t = %.1f) around a cylinder", t),
                         xlabel = "x",
                         ylabel = "y",
                         levels = v_levels,
                          xlims = (grid.xF[1], grid.xF[grid.Nx]),
                          ylims = (grid.yF[1], grid.yF[grid.Ny]),
                          clims = (-v_lim, v_lim))
    plot!(circle_shape(30,20,1),seriestype=[:shape,],linecolor=:black,
    legend=false,fillalpha=0, aspect_ratio=1)
end

gif(anim, "flow_around_cyl_velocity_tempIBM_newBCs_51.gif", fps = 8) # hide

@info "Making a neat movie of temperature..."

anim = @animate for (i, iteration) in enumerate(iterations)

    @info "Plotting frame $i from iteration $iteration..."
    
    t = file["timeseries/t/$iteration"]

    v_slice = file["timeseries/T/$iteration"][:, :, 1]
    @info maximum(v_slice) minimum(v_slice)
    v_max = maximum(abs, v_slice)
    v_lim = 0.8 * v_max

    v_levels = vcat([-v_max], range(-v_lim, stop=v_lim, length=50), [v_max])

    v_plot = contourf(grid.xC[1:500], grid.yC[1:750], v_slice';
                      linewidth = 0,
                          color = :heat,
                    aspectratio = 1,
                          title = @sprintf("T(x, y, t = %.1f) around a cylinder", t),
                         xlabel = "x",
                         ylabel = "y",
                         levels = v_levels,
                          xlims = (grid.xC[1], grid.xC[grid.Nx]),
                          ylims = (grid.yC[1], grid.yC[grid.Ny]),
                          clims = (-v_lim, v_lim))
    plot!(circle_shape(30,20,1),seriestype=[:shape,],linecolor=:black,
    legend=false,fillalpha=0, aspect_ratio=1)
end

gif(anim, "flow_around_cyl_tempIBM_newBCs_51.gif", fps = 8) # hide

