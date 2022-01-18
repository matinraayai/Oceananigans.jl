using Oceananigans.Grids: halo_size
using Oceananigans.AbstractOperations: GridMetricOperation, XAreaMetric, YAreaMetric
# Has to be changed when the regression data is updated 

@kernel function _compute_vertically_integrated_lateral_areas!(∫ᶻ_A, grid)
    i, j = @index(Global, NTuple)

    @inbounds begin
        ∫ᶻ_A.xᶠᶜᶜ[i, j, 1] = 0
        ∫ᶻ_A.yᶜᶠᶜ[i, j, 1] = 0

        @unroll for k in 1:grid.Nz
            ∫ᶻ_A.xᶠᶜᶜ[i, j, 1] += Δyᶠᶜᵃ(i, j, k, grid) * Δzᶠᶜᶜ(i, j, k, grid)
            ∫ᶻ_A.yᶜᶠᶜ[i, j, 1] += Δxᶜᶠᵃ(i, j, k, grid) * Δzᶜᶠᶜ(i, j, k, grid)
        end
    end
end

function compute_vertically_integrated_lateral_areas!(∫ᶻ_A)

    # we have to account for halos when calculating Integrated areas, in case 
    # a periodic domain, where it is not guaranteed that ηₙ == ηₙ₊₁ 
    # 2 halos (instead of only 1) are necessary to accomodate the preconditioner

    field_grid = ∫ᶻ_A.xᶠᶜᶜ.grid

    Ax_fcc = GridMetricOperation((Face, Center, Center), XAreaMetric(), field_grid)
    Ay_cfc = GridMetricOperation((Center, Face, Center), YAreaMetric(), field_grid)

    sum!(∫ᶻ_A.xᶠᶜᶜ, Ax_fcc)
    sum!(∫ᶻ_A.yᶜᶠᶜ, Ay_cfc)

    fill_halo_regions!(∫ᶻ_A.xᶠᶜᶜ, arch)
    fill_halo_regions!(∫ᶻ_A.yᶜᶠᶜ, arch)

    return nothing
end
