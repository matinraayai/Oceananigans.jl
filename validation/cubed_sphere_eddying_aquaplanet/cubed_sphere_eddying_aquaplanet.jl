using Statistics
using Logging
using Printf
using SpecialFunctions
using DataDeps
using JLD2

using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary

using Dates: now, Second, format
using Oceananigans.Models.HydrostaticFreeSurfaceModels: VerticalVorticityField
using Oceananigans.Diagnostics: accurate_cell_advection_timescale
using Oceananigans.CubedSpheres: CubedSphereFaces, inject_cubed_sphere_exchange_boundary_conditions

Logging.global_logger(OceananigansLogger())

#####
##### Progress monitor
#####

mutable struct Progress
    interval_start_time :: Float64
end

function (p::Progress)(sim)
    wall_time = (time_ns() - p.interval_start_time) * 1e-9
    progress = sim.model.clock.time / sim.stop_time
    ETA = (1 - progress) / progress * sim.run_time
    ETA_datetime = now() + Second(round(Int, ETA))

    @info @sprintf("[%06.2f%%] Time: %s, iteration: %d, max(|u⃗|): (%.2e, %.2e) m/s, extrema(η): (min=%.2e, max=%.2e), CFL: %.2e",
                   100 * progress,
                   prettytime(sim.model.clock.time),
                   sim.model.clock.iteration,
                   maximum(abs, sim.model.velocities.u),
                   maximum(abs, sim.model.velocities.v),
                   minimum(sim.model.free_surface.η),
                   maximum(sim.model.free_surface.η),
                   sim.parameters.cfl(sim.model))

    @info @sprintf("           ETA: %s (%s), Δ(wall time): %s / iteration",
                   format(ETA_datetime, "yyyy-mm-dd HH:MM:SS"),
                   prettytime(ETA),
                   prettytime(wall_time / sim.iteration_interval))

    p.interval_start_time = time_ns()

    return nothing
end

#####
##### Loading grids
#####

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

Logging.global_logger(OceananigansLogger())

dd32 = DataDep("cubed_sphere_32_grid",
    "Conformal cubed sphere grid with 32×32 grid points on each face",
    "https://github.com/CliMA/OceananigansArtifacts.jl/raw/main/cubed_sphere_grids/cubed_sphere_32_grid.jld2",
    "b1dafe4f9142c59a2166458a2def743cd45b20a4ed3a1ae84ad3a530e1eff538" # sha256sum
)

dd96 = DataDep("cubed_sphere_96_grid",
    "Conformal cubed sphere grid with 96×96 grid points on each face",
    "https://github.com/CliMA/OceananigansArtifacts.jl/raw/main/cubed_sphere_grids/cs96/cubed_sphere_96_grid.jld2"
)

DataDeps.register(dd32)
DataDeps.register(dd96)

#####
##### Utils
#####

function diagnose_velocities_from_streamfunction(ψ, grid)
    ψᶠᶠᶜ = Field(Face, Face,   Center, CPU(), grid)
    uᶠᶜᶜ = Field(Face, Center, Center, CPU(), grid)
    vᶜᶠᶜ = Field(Center, Face, Center, CPU(), grid)

    for (f, grid_face) in enumerate(grid.faces)
        Nx, Ny, Nz = size(grid_face)

        ψᶠᶠᶜ_face = ψᶠᶠᶜ.data.faces[f]
        uᶠᶜᶜ_face = uᶠᶜᶜ.data.faces[f]
        vᶜᶠᶜ_face = vᶜᶠᶜ.data.faces[f]

        for i in 1:Nx+1, j in 1:Ny+1
            ψᶠᶠᶜ_face[i, j, 1] = ψ(grid_face.λᶠᶠᵃ[i, j], grid_face.φᶠᶠᵃ[i, j])
        end

        for i in 1:Nx+1, j in 1:Ny
            uᶠᶜᶜ_face[i, j, 1] = (ψᶠᶠᶜ_face[i, j, 1] - ψᶠᶠᶜ_face[i, j+1, 1]) / grid.faces[f].Δyᶠᶜᵃ[i, j]
        end

        for i in 1:Nx, j in 1:Ny+1
            vᶜᶠᶜ_face[i, j, 1] = (ψᶠᶠᶜ_face[i+1, j, 1] - ψᶠᶠᶜ_face[i, j, 1]) / grid.faces[f].Δxᶜᶠᵃ[i, j]
        end
    end

    return uᶠᶜᶜ, vᶜᶠᶜ, ψᶠᶠᶜ
end

function cubed_sphere_surface_momentum_flux_bcs(τx, τy, grid; ε=0.01)

    for f in 1:length(grid.faces)
        @. τx.data.faces[f] += ε * randn()
        @. τy.data.faces[f] += ε * randn()
    end

    Nx, Ny, Nz = size(grid.faces[1])

    u_bcs_faces = Tuple(
        inject_cubed_sphere_exchange_boundary_conditions(
            FieldBoundaryConditions(face_grid, (Face, Center, Center); top=FluxBoundaryCondition(τx.data.faces[face_number][1:Nx+1, 1:Ny, 1])),
            face_number,
            grid.face_connectivity
        )
        for (face_number, face_grid) in enumerate(grid.faces)
    )

    v_bcs_faces = Tuple(
        inject_cubed_sphere_exchange_boundary_conditions(
            FieldBoundaryConditions(face_grid, (Center, Face, Center); top=FluxBoundaryCondition(τy.data.faces[face_number][1:Nx, 1:Ny+1, 1])),
            face_number,
            grid.face_connectivity
        )
        for (face_number, face_grid) in enumerate(grid.faces)
    )

    u_bcs = CubedSphereFaces{typeof(u_bcs_faces[1]), typeof(u_bcs_faces)}(u_bcs_faces)
    v_bcs = CubedSphereFaces{typeof(v_bcs_faces[1]), typeof(v_bcs_faces)}(v_bcs_faces)

    return u_bcs, v_bcs
end

#####
##### Script starts here
#####

function cubed_sphere_eddying_aquaplanet(grid_filepath)

    ## Grid setup

    H = 100meters
    underlying_grid = ConformalCubedSphereGrid(grid_filepath, Nz=1, z=(-H, 0))

    ## Read in a depth mask
    bfile="bathy_cs32.bin"
    bfile="bathy_Hmin50.bin"
    depths=reshape(bswap.( reinterpret(Float64, read(bfile,sizeof(Float64)*32*32*6 ) ) ),(32,6,32))
    dfaces=(
     depths[:,1,:],
     depths[:,2,:],
     depths[:,3,:],
     depths[:,4,:],
     depths[:,5,:],
     depths[:,6,:]
    )
    function solid(x, y, z, i, j, k, face_number, grid)
     is_solid = false
     ## Need to fix dfaces to be proper cube field with halo
     if i > 0 && j > 0
      if dfaces[face_number][i,j] > -H
        is_solid = true
      end
     end
     ## println( typeof() )

     ## depths[i,j] > -z 

     return is_solid
    end

    csib= (
            (i,j,z) -> dfaces[1][i,j] > -H,
            (i,j,z) -> dfaces[2][i,j] > -H,
            (i,j,z) -> dfaces[3][i,j] > -H,
            (i,j,z) -> dfaces[4][i,j] > -H,
            (i,j,z) -> dfaces[5][i,j] > -H,
            (i,j,z) -> dfaces[6][i,j] > -H
          )
    solid6 = (
               (x,y,z,i,j,k,fnum) -> i > 0 && j > 0 ? csib[1](i,j,z) : false,
               (x,y,z,i,j,k,fnum) -> i > 0 && j > 0 ? csib[2](i,j,z) : false,
               (x,y,z,i,j,k,fnum) -> i > 0 && j > 0 ? csib[3](i,j,z) : false,
               (x,y,z,i,j,k,fnum) -> i > 0 && j > 0 ? csib[4](i,j,z) : false,
               (x,y,z,i,j,k,fnum) -> i > 0 && j > 0 ? csib[5](i,j,z) : false,
               (x,y,z,i,j,k,fnum) -> i > 0 && j > 0 ? csib[6](i,j,z) : false
             )


    #=
    cubed_sphere_immersed_boundary = (GridFittedTopography(depths[:, 1, :]),
				      GridFittedTopography(depths[:, 2, :]),
				      GridFittedTopography(depths[:, 3, :]),
				      GridFittedTopography(depths[:, 4, :]),
				      GridFittedTopography(depths[:, 5, :]),
				      GridFittedTopography(depths[:, 6, :]))
 
    grid = ImmersedBoundaryGrid(underlying_grid, cubed_sphere_immersed_boundary)
    =#

    # Leave this here for closure mess for now - this will be a little wrong, but the fix
    # looks to me like it needs some face number upgrade. One feature could be a parent
    # patch rank held with a grid? This could avoid need for an arg? It could potentially
    # also invovle a depth so that grid patching could be multi-level, but still efficcient?
    solid(x, y, z) = false
    grid = underlying_grid

    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBoundary(solid6))

    ## "Tradewind-like" zonal wind stress pattern where -π/2 ≤ φ ≤ π/2
    τx(φ) = 0.1 * exp(-10 * (φ - π/3)^2) + 0.1 * exp(-10 * (φ + π/3)^2) - 0.2 * exp(-8φ^2) + 0.19 * exp(-20φ^2)

    coriolis = HydrostaticSphericalCoriolis(scheme=VectorInvariantEnstrophyConserving())
    R = grid.faces[1].radius
    Ω = coriolis.rotation_rate

    # Streamfunction integrated using Mathematica
    # Scale by Ω^2 * R^2 * sin(φ) to get units of momentum flux (m²/s²) on the sphere.
    # Minus sign because a negative flux drives currents in the positive x-direction.
    # Hmmm, I think I messed up the magnitude but the resulting wind stress patterns should do the trick for now. It's a little too strong (stronger than the 0.1 N/m² we expect from τx).
    # I'm further scaling it down by a factor so the wind stress is even weaker (for stability/debugging).
    ψ̃(λ, φ) = - 1e-4 * Ω^2 * R^2 * sin(φ) * (-0.02802495608 * erf(1.054092553 * (π - 3φ)) - 0.06266570687 * erf(2.828427125φ) + 0.03765160933 * erf(4.472135955φ) + 0.02802495608 * erf(1.054092553 * (π + 3φ)))

    # Ocean scaling for typical ∂u/∂t from wind
    # Hʳᵉᶠ - reference depth (m)
    # ρʳᵉᶠ - reference denisty (kg m^-3 )
    # τ_scale_fac - scaling for factors that don't make sense yet
    # I am not sure this makes sense yet. The BC code is formulated for ms^-1 . m^-2
    # so something seems a little off?
    τʳᵉᶠ=0.1
    Hʳᵉᶠ=4000
    ρʳᵉᶠ=1000
    τ_scale_fac=150
    scale_fac = τʳᵉᶠ/Hʳᵉᶠ/ρʳᵉᶠ*τ_scale_fac
    ψ̃(λ, φ) = -R*scale_fac*(-0.02802495608 * erf(1.054092553 * (π - 3φ)) - 0.06266570687 * erf(2.828427125φ) + 0.03765160933 * erf(4.472135955φ) + 0.02802495608 * erf(1.054092553 * (π + 3φ)))

    λ̃(λ) = (λ + 180)/ 360 * 2π
    φ̃(φ) = φ / 180 * π
    ψ(λ, φ) = ψ̃(λ̃(λ), φ̃(φ))

    u_top_flux, v_top_flux, ψ₀ = diagnose_velocities_from_streamfunction(ψ, grid)

    u_bcs, v_bcs = cubed_sphere_surface_momentum_flux_bcs(u_top_flux, v_top_flux, grid, ε=0)

    # Linear damping so the wind stress doesn't keep accelerating the fluid.
    # @inline linear_damping(λ, φ, z, t, u, μ) = - μ * u

    # μ = 1e-3
    # u_forcing = Forcing(linear_damping, parameters=μ, field_dependencies=:u)
    # v_forcing = Forcing(linear_damping, parameters=μ, field_dependencies=:v)

    # Since the continuous forcing seems to slow down the model by a factor of ~2x.
    @inline linear_damping_u(i, j, k, grid, clock, model_fields, μ) = @inbounds - μ * model_fields.u[i, j, k]
    @inline linear_damping_v(i, j, k, grid, clock, model_fields, μ) = @inbounds - μ * model_fields.v[i, j, k]

    μ = 1/year
    u_forcing = Forcing(linear_damping_u, parameters=μ, discrete_form=true)
    v_forcing = Forcing(linear_damping_v, parameters=μ, discrete_form=true)

    ## Model setup

    model = HydrostaticFreeSurfaceModel(
               architecture = CPU(),
                       grid = grid,
         momentum_advection = VectorInvariant(),
               free_surface = ExplicitFreeSurface(gravitational_acceleration=0.5),
#                  coriolis = nothing,
                   coriolis = HydrostaticSphericalCoriolis(scheme=VectorInvariantEnstrophyConserving()),
#                   closure = HorizontallyCurvilinearAnisotropicDiffusivity(νh=5000),
                    closure = nothing,
        boundary_conditions = (u=u_bcs, v=v_bcs),
                  # forcing = (u=u_forcing, v=v_forcing),
                    tracers = nothing,
                   buoyancy = nothing
    )

    # Some random noise to get things going.
    ε(λ, φ, z) = 0.1 * randn()
    Oceananigans.set!(model, u=ε, v=ε)

    ## Simulation setup

    Δt = 40minutes

    g = model.free_surface.gravitational_acceleration
    gravity_wave_speed = √(g * H)
    min_spacing = filter(!iszero, grid.faces[1].Δyᶠᶠᵃ) |> minimum
    wave_propagation_time_scale = min_spacing / gravity_wave_speed
    gravity_wave_cfl = Δt / wave_propagation_time_scale
    @info @sprintf("Gravity wave CFL = %.4f", gravity_wave_cfl)

    deformation_radius_45°N = √(g * H) / (2Ω*sind(45))
    @info @sprintf("Deformation radius @ 45°N: %.2f km", deformation_radius_45°N / 1000)

    cfl = CFL(Δt, accurate_cell_advection_timescale)

    simulation = Simulation(model,
                        Δt = Δt,
##               stop_time = 5years,
                 stop_time = Δt*100,
        iteration_interval = 20,
                  progress = Progress(time_ns()),
                parameters = (; cfl)
    )

    output_fields = merge(model.velocities, (η=model.free_surface.η, ζ=VerticalVorticityField(model)))

    simulation.output_writers[:fields] =
        JLD2OutputWriter(model, output_fields,
            schedule = TimeInterval(1day),
              prefix = "cubed_sphere_eddying_aquaplanet",
               force = true)

    run!(simulation)

    return simulation
end

## include("animate_on_map_projection.jl")

function run_cubed_sphere_eddying_aquaplanet()

    # simulation = cubed_sphere_eddying_aquaplanet(datadep"cubed_sphere_96_grid/cubed_sphere_96_grid.jld2")
    simulation = cubed_sphere_eddying_aquaplanet(datadep"cubed_sphere_32_grid/cubed_sphere_32_grid.jld2")

    ## projections = [
    ##     ccrs.NearsidePerspective(central_longitude=0, central_latitude=30),
    ##     ccrs.NearsidePerspective(central_longitude=180, central_latitude=-30)
    ## ]

    # animate_eddying_aquaplanet(projections=projections)

    return simulation
end
