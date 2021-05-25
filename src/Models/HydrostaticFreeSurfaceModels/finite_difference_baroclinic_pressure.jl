using Oceananigans.Operators: Δzᵃᵃᶜ
using Oceananigans.Buoyancy: buoyancy_perturbation

struct FiniteDifferencePressureGradientScheme end

function KinematicBaroclinicPressure(scheme::S, arch, grid) where S <: FiniteDifferencePressureGradientScheme
    p = CenterField(arch, grid, TracerBoundaryConditions(grid))
    return KinematicBaroclinicPressure{S}(p)
end

# Default
KinematicBaroclinicPressure(scheme::Nothing, arch, grid) =
    KinematicBaroclinicPressure(FiniteDifferencePressureGradientScheme(), arch, grid)

const FiniteDifferencePressure = KinematicBaroclinicPressure{<:FiniteDifferencePressureGradientScheme}

"""
Update the hydrostatic pressure perturbation pHY′. This is done by integrating
the `buoyancy_perturbation` downwards:

    `pHY′ = ∫ buoyancy_perturbation dz` from `z=0` down to `z=-Lz`
"""
@kernel function _calculate_kinematic_baroclinic_pressure!(pressure::FiniteDifferencePressure, grid, buoyancy, tracers)
    i, j = @index(Global, NTuple)

    p = pressure.p 
    Nz = grid.Nz

    # Surface-adjacent
    @inbounds p[i, j, Nz] = - ℑzᵃᵃᶠ(i, j, Nz+1, grid, buoyancy_perturbation, buoyancy, tracers) * Δzᵃᵃᶠ(i, j, Nz+1, grid)

    @unroll for k in Nz-1 : -1 : 1
        @inbounds p[i, j, k] = p[i, j, k+1] - ℑzᵃᵃᶠ(i, j, k+1, grid, buoyancy_perturbation, buoyancy, tracers) * Δzᵃᵃᶠ(i, j, k+1, grid)
    end
end

baroclinic_pressure_x_gradient(i, j, k, grid, baroclinic_pressure::FiniteDifferencePressure) = ∂xᶠᶜᵃ(i, j, k, grid, baroclinic_pressure.p)
baroclinic_pressure_y_gradient(i, j, k, grid, baroclinic_pressure::FiniteDifferencePressure) = ∂yᶜᶠᵃ(i, j, k, grid, baroclinic_pressure.p)

