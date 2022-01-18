using Oceananigans.Grids: halo_size
using Oceananigans.AbstractOperations: Ax, Ay, GridMetricOperation
# Has to be changed when the regression data is updated 

function compute_vertically_integrated_lateral_areas!(∫ᶻ_A)

    # we have to account for halos when calculating Integrated areas, in case 
    # a periodic domain, where it is not guaranteed that ηₙ == ηₙ₊₁ 
    # 2 halos (instead of only 1) are necessary to accomodate the preconditioner

    field_grid = ∫ᶻ_A.xᶠᶜᶜ.grid
    arch = architecture(field_grid)

    Axᶠᶜᶜ = GridMetricOperation((Face, Center, Center), Ax, field_grid)
    Ayᶜᶠᶜ = GridMetricOperation((Center, Face, Center), Ay, field_grid)
    
    sum!(∫ᶻ_A.xᶠᶜᶜ, Axᶠᶜᶜ)
    sum!(∫ᶻ_A.yᶜᶠᶜ, Ayᶜᶠᶜ)

    fill_halo_regions!(∫ᶻ_A.xᶠᶜᶜ, arch)
    fill_halo_regions!(∫ᶻ_A.yᶜᶠᶜ, arch)

    return nothing
end
