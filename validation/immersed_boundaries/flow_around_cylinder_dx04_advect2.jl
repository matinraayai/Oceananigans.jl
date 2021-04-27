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
topology=(Periodic, Bounded, Bounded)

# setting up 2D grid
grid = RegularRectilinearGrid(topology=topology, size=(500, 500, 1), x=(20, 40), y=(10, 30), z=(0, 1))


# reynolds number
Re = 40

#cylinder with center at (30,20)
const R = 1 # radius
vCent = [30.0, 20.0, 0.];
dist_cylinder(v) = sqrt((vCent[1]-v[1])^2+(vCent[2]-v[2])^2)-R # immersed solid
inside_cylinder(x, y, z) = ((x-vCent[1])^2 + (y-vCent[2])^2) <= R # immersed solid

# boundary conditions: inflow and outflow in y
v_bcs = VVelocityBoundaryConditions(grid,
                                    north = BoundaryCondition(NormalFlow,1.0),
                                    south = BoundaryCondition(NormalFlow,1.0))

# setting up incompressible model with immersed boundary
model = IncompressibleModel(timestepper = :RungeKutta3, 
                              advection = CenteredSecondOrder(),
                                   grid = grid,
                               buoyancy = nothing,
                                tracers = nothing,
                                closure = IsotropicDiffusivity(ν=1/Re),
                    boundary_conditions = (v=v_bcs,),
                      immersed_boundary = dist_cylinder
                           )

# initial condition
# setting velocitiy to zero inside the cylinder and 1 everywhere else
v₀(x, y, z) = ifelse(inside_cylinder(x,y,z), 0., 1.)
set!(model, v=v₀)


progress(sim) = @info @sprintf("Iteration: % 4d, time: %.2f, max(v): %.5f, min(v): %.5f",
sim.model.clock.iteration,
sim.model.clock.time,
maximum(sim.model.velocities.v.data),
minimum(sim.model.velocities.v.data))

simulation = Simulation(model, Δt=5.7e-3, stop_time=60, iteration_interval=100, progress=progress)

# ## Output

simulation.output_writers[:fields] = JLD2OutputWriter(model,
                                                      merge(model.velocities, model.pressures),
                                                      schedule = TimeInterval(1.0),
                                                      prefix = "flow_around_cylinder_dx04_advect2",
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

gif(anim, "flow_around_cyl_velocity_dx04_advect2.gif", fps = 8) # hide
