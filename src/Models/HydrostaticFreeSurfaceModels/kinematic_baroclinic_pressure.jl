using Adapt

struct KinematicBaroclinicPressure{S, P}
    scheme :: S
    p :: P
end

# Fallback
KinematicBaroclinicPressure(pressure::KinematicBaroclinicPressure, arch, grid) = pressure

Adapt.adapt_structure(to, pressure::KinematicBaroclinicPressure) =
    KinematicBaroclinicPressure(pressure.scheme, adapt(to, pressure.p))

include("finite_difference_baroclinic_pressure.jl")
include("contact_force_baroclinic_pressure.jl")

calculate_kinematic_baroclinic_pressure!(model) = calculate_kinematic_baroclinic_pressure!(model.baroclinic_pressure.p,
                                                                                           model.architecture,
                                                                                           model.grid,
                                                                                           model.buoyancy,
                                                                                           model.tracers)
    

function calculate_kinematic_baroclinic_pressure!(pressure, arch, grid, buoyancy, tracers)

    pressure_calculation = launch!(arch, grid, Val(:xy),
                                   _calculate_kinematic_baroclinic_pressure!, pressure, grid, buoyancy, tracers,
                                   dependencies = device_event(model))

    # Fill halo regions for pressure
    wait(device(arch), pressure_calculation)

    fill_halo_regions!(pressure.p, arch)

    return nothing
end



