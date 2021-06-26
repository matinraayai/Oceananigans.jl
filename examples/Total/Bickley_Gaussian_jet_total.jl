using Oceananigans
using Oceananigans.Units: hours, days, seconds, meters, kilometers, minutes
using NCDatasets, Plots, Printf
using LinearAlgebra: norm

function norm_ũ(model)
   compute!(ũ)
   return norm(interiorparent(ũ))
end

function norm_ṽ(model)
   compute!(ṽ)
   return norm(interiorparent(ṽ)) 
end

function norm_b̃(model)
   compute!(b̃)
   return norm(interiorparent(b̃))
end

   const Ly = 1000kilometers
   const Lz = 2.5kilometers
    const D = Lz/2
const L_jet = Ly/10

const θ₀ =  π/32
Coriolis = NonTraditionalFPlane(latitude=θ₀*180/π)

const N² = 1e-4      # change to 1e-4
const U_max = 14.6

const νz = 0.26 #1.27e-2

grid = RegularRectilinearGrid(size = (256, 256),
                                 y = (-Ly/2, Ly/2), z=(-Lz, 0),
                          topology = (Flat, Bounded, Bounded),
                              halo = (3, 3))

B_func(x, y, z, t, N) = N² * (z + D)
                    N = sqrt(N²)
                    B = BackgroundField(B_func, parameters=N)

# Jet Profile:
# Bickley x Gaussian [horizontal x vertical]
ū(x, y, z) = U_max * sech(y/L_jet)^2
b̄(x, y, z) = - Coriolis.fy * ū(x, y, z)

perturbation(x, y, z) = randn() * sech(y/L_jet)^2
          uⁱ(x, y, z) = ū(x, y, z) + 1e-2 * perturbation(x, y, z)
          bⁱ(x, y, z) = b̄(x, y, z) + 1e-6 * perturbation(x, y, z)

#u_forcing_func(x, y, z, t, ν) = 2 * νz / H_jet^2 * ū(x, y, z) * ( 1 - 2/H_jet^2 * (z + D)^2)
#u_forcing = Forcing(u_forcing_func, parameters = νz)

model = IncompressibleModel(
       architecture = CPU(),
               grid = grid,
#          advection = WENO5(),
        timestepper = :RungeKutta3,
           coriolis = Coriolis,
            tracers = :b,
 background_fields = (b=B,),
           buoyancy = BuoyancyTracer(),
            closure = AnisotropicDiffusivity(νh=0, νz=νz),
#            forcing = (u = u_forcing, )
            )

set!(model, u = uⁱ, b = bⁱ)

u = model.velocities.u
v = model.velocities.v
b = model.tracers.b

ũ = ComputedField(u - ū)
ṽ = ComputedField(v)
b̃ = ComputedField(b - b̄)

y, z   = ynodes(model.velocities.u), znodes(model.velocities.u)
yv, zv = ynodes(model.velocities.v), znodes(model.velocities.v)
yb, zb = ynodes(model.tracers.b), znodes(model.tracers.b)

kwargs = (
            xlabel= "y (km)",
            ylabel= "z (km)",
         linewidth= 0,
             color=:balance,
          colorbar= true,
             xlims= (-Ly/2e3, Ly/2e3),
             ylims= (-Lz/1e3, 0)
         )
progress(sim) = @printf("Iteration: %d, time: %s, Δt: %s\n",
                        sim.model.clock.iteration,
                        prettytime(sim.model.clock.time),
                        prettytime(sim.Δt))

#wizard = TimeStepWizard(cfl=1.0, Δt=10minutes, max_change=1.1, max_Δt=2minutes)
simulation = Simulation(model, Δt=60, stop_time=1hours, #17days,
                        iteration_interval=60, progress=progress)

simulation.output_writers[:fields] =
   NetCDFOutputWriter(
          model,
            (u = ũ, v = ṽ, b = b̃),
      filepath = joinpath(@__DIR__, "inertially_unstable_jet_fields.nc"),
      schedule = IterationInterval(60),
          mode = "c")

simulation.output_writers[:norms] =
   NetCDFOutputWriter(
            model,
         (norm_u = norm_ũ, norm_v = norm_ṽ, norm_b = norm_b̃),
        filepath = joinpath(@__DIR__, "inertially_unstable_jet_norms.nc"),
        schedule = IterationInterval(1),
      dimensions = (norm_u=(), norm_v=(), norm_b=()),
            mode = "c")

start_time = time_ns()
run!(simulation)
finish_time = time_ns()
print("Simulation time = ", prettytime(finish_time - start_time), "\n")

ds = NCDataset(simulation.output_writers[:fields].filepath, "r")

iterations = keys(ds["time"])

@info "Making a movie of perturbation zonal velocity and buoyancy..."

anim = @animate for (iter, t) in enumerate(ds["time"])

   @info "Plotting frame $iter from time $t..."

   u_snapshot = ds["u"][1, :, :, iter]
   v_snapshot = ds["v"][1, :, :, iter]
   b_snapshot = ds["b"][1, :, :, iter]


   u_max = maximum(abs, u_snapshot)
   u_plot = contourf(y/1e3, z/1e3, u_snapshot',
                     title = @sprintf("ũ at t = %.1f hours", t/hours),
                     color=:balance,
                     clim=(-u_max, u_max);
                     kwargs...)

   v_max = maximum(abs, v_snapshot)
   v_plot = contourf(y/1e3, z/1e3, v_snapshot',
                     title = @sprintf("ṽ at t = %.1f hours", t/hours),
                     color=:balance,
                     clim=(-v_max, v_max);
                     kwargs...)

   b_max = maximum(abs, b_snapshot)
   b_plot = contourf(y/1e3, z/1e3, b_snapshot',
                     title = @sprintf("b̃ at t = %.1f hours", t/hours),
                     color=:balance;
                     clim=(-b_max, b_max),
                     kwargs...)

    plt = plot(u_plot, v_plot, b_plot, layout=(1, 2), size=(1200, 500))

end

close(ds)

mp4(anim, "Inertial_Instability_2D.mp4", fps=15)

ds2 = NCDataset(simulation.output_writers[:norms].filepath, "r")

iterations = keys(ds2["time"])

     t = ds2["time"][:]
norm_u = ds2["norm_u"][:]
norm_v = ds2["norm_v"][:]
norm_b = ds2["norm_b"][:]

close(ds2)

#plt = plot(t, norm_u, label="u", title="Norms")
#plot!(plt, t, norm_b, label="b")

using Polynomials: fit

#### change this!!!!
I = 20000:21000

degree = 1
linear_fit_polynomial = fit(t[I], log.(norm_u[I]), degree, var = :t)

constant, slope = linear_fit_polynomial[0], linear_fit_polynomial[1]

best_fit = @. exp(constant + slope * t)

plt = plot(t/hours, norm_u,
        yaxis = :log,
        #ylims = (1, 500),
           lw = 4,
        label = "norm(u)",
       xlabel = "time (hours)",
       ylabel = "norm(u)",
        title = "norm of perturbation",
       legend = :bottomright)

plot!(plt, t[I]/hours, 2 * best_fit[I], # factor 2 offsets fit from curve for better visualization
           lw = 4,
        label = "best fit")

savefig(plt, "growth.png")

# To-Do
# 2. Try wizard
# 4. Do 2D and 3D simulations on cedar (cpu and then cpu)
# 5. Try MPI?


# To-Do this week
# 1. Switch sign of U in linear stability to see if we get the same shapes
# 2. Compute growth rates and compare with theory
# 3. Try N = 1e-3
# 4. Try N = 1e-4
