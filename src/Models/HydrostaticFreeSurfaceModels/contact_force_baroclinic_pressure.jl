struct ContactForcePressureGradientScheme end

function KinematicBaroclinicPressure(scheme::S, arch, grid) where S <: ContactForcePressureGradientScheme
    pressure_bcs = FieldBoundaryConditions(grid, (Center, Center, Face), top = nothing, bottom = nothing)
    p = ZFaceField(arch, grid, pressure_bcs)
    return KinematicBaroclinicPressure{S}(p)
end

const ContactForcePressure = KinematicBaroclinicPressure{<:ContactForcePressureGradientScheme}

"""
Update the baroclinic component of the kinematic pressure by integrating
the `buoyancy_perturbation` downwards:

    `p = ∫ buoyancy_perturbation dz` from `z=0` down to `z=-Lz`

noting that `p` is located at `(Center, Center, Face)`.
"""
@kernel function _calculate_kinematic_baroclinic_pressure!(pressure::ContactForcePressure, grid, buoyancy, tracers)
    i, j = @index(Global, NTuple)

    p = pressure.p 
    Nz = grid.Nz

    # Surface pressure set to zero for now. Later could fill top halo prior to launch.
    @inbounds p[i, j, Nz+1] = 0

    @unroll for k in Nz : -1 : 1
        @inbounds p[i, j, k] = p[i, j, k+1] - buoyancy_perturbation(i, j, k, grid, buoyancy, tracers) * Δzᵃᵃᶜ(i, j, k, grid)
    end
end

# Move to Grids module eventually...
@inline zᵃᵃᶠ(i, j, k, grid) = znode(Center(), Center(), Face(), i, j, k, grid)

@inline function baroclinic_pressure_x_gradient(i, j, k, grid, baroclinic_pressure::ContactForcePressure)
    p = baroclinic_pressure.p

    # Integrate V⁻¹ ∬ ∂x p dV = Ay⁻¹ ∮ p n̂ ⋅ x̂ dℓ = ∫ p dz |west - ∫ p dz |top - ∫ p dz |east + ∫ p dz |bottom
    west   = ℑzᵃᵃᶜ(i, j, k, grid, p)   * Δzᵃᵃᶜ(i, j, k,   grid)
    east   = ℑzᵃᵃᶜ(i+1, j, k, grid, p) * Δzᵃᵃᶜ(i, j, k,   grid)
    top    = ℑxᵃᵃᶠ(i, j, k+1, grid, p) * δxᶠᵃᵃ(i, j, k+1, grid, zᵃᵃᶠ) 
    bottom = ℑxᵃᵃᶠ(i, j, k, grid, p)   * δxᶠᵃᵃ(i, j, k,   grid, zᵃᵃᶠ) 

    # note inwards-pointing normal and uniform-in-y assumption
    return (west - top - east + bottom) / Ayᶠᶜᶜ(i, j, k, grid)
end

@inline function baroclinic_pressure_y_gradient(i, j, k, grid, baroclinic_pressure::ContactForcePressure)
    p = baroclinic_pressure.p

    # Integrate V⁻¹ ∬ ∂y p d V = Ax⁻¹ ∮ p n̂ ⋅ ŷ dℓ = ∫ p dz |south - ∫ p dz |top - ∫ p dz |north + ∫ p dz |bottom
    south  = ℑzᵃᵃᶜ(i, j, k, grid, p)   * Δzᵃᵃᶜ(i, j, k,   grid)
    north  = ℑzᵃᵃᶜ(i, j+1, k, grid, p) * Δzᵃᵃᶜ(i, j, k,   grid)
    top    = ℑxᵃᵃᶠ(i, j, k+1, grid, p) * δyᵃᶠᵃ(i, j, k+1, grid, zᵃᵃᶠ) 
    bottom = ℑxᵃᵃᶠ(i, j, k, grid, p)   * δyᵃᶠᵃ(i, j, k,   grid, zᵃᵃᶠ) 

    # note inwards-pointing normal and uniform-in-x assumption
    return (south - top - north + bottom) / Axᶜᶠᶜ(i, j, k, grid)
end
