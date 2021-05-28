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
grid = RegularRectilinearGrid(topology=topology, size=(350, 350, 1), x=(20, 40), y=(10, 30), z=(0, 1), halo=(3, 3, 3))


# reynolds number
Re = 100

#cylinder with center at (30,20)
const R = 1 # radius
vCent = [30.0, 20.0, 0.];
dist_cylinder(v) = sqrt((vCent[1]-v[1])^2+(vCent[2]-v[2])^2)-R # immersed solid
inside_cylinder(x, y, z) = ((x-vCent[1])^2 + (y-vCent[2])^2) <= R # immersed solid


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
                      immersed_boundary = dist_cylinder
                           )

# initial condition
# setting velocitiy to zero inside the cylinder and 1 everywhere else
v₀(x, y, z) = ifelse(inside_cylinder(x,y,z), 0., 1.)
T₀(x, y, z) = 0. # temperature zero at first
set!(model, v=v₀, T=T₀)


progress(sim) = @info @sprintf("Iteration: % 4d, time: %.2f, max(v): %.5f, min(v): %.5f",
sim.model.clock.iteration,
sim.model.clock.time,
maximum(sim.model.velocities.v.data),
minimum(sim.model.velocities.v.data))

simulation = Simulation(model, Δt=5.7e-3, stop_time=50, iteration_interval=100, progress=progress)

# ## Output

simulation.output_writers[:fields] = JLD2OutputWriter(model,
                                                      merge(model.velocities, model.tracers),
                                                      schedule = TimeInterval(1.0),
                                                      prefix = "flow_around_cylinder_tempIBM_150s_Re100",
                                                      force = true)

# run it
run!(simulation)
