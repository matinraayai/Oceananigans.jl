using Oceananigans.AbstractOperations: Ax, Ay, GridMetricOperation

"""
Compute the vertical integrated volume flux from the bottom to ``z=0`` (i.e., linear free-surface).

```
U★ = ∫ᶻ Ax * u★ dz
V★ = ∫ᶻ Ay * v★ dz
```
"""
### Note - what we really want is RHS = divergence of the vertically integrated volume flux
###        we can optimize this a bit later to do this all in one go to save using intermediate variables.
function compute_vertically_integrated_volume_flux!(∫ᶻ_U, model)

    # Fill halo regions for predictor velocity.
    fill_halo_regions!(model.velocities, model.architecture, model.clock, fields(model))

    sum!(∫ᶻ_U.u, Ax * model.velocities.u)
    sum!(∫ᶻ_U.v, Ay * model.velocities.v)

    # We didn't include right boundaries, so...
    fill_halo_regions!(∫ᶻ_U, model.architecture, model.clock, fields(model))

    return nothing
end
