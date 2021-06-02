using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!, mask_immersed_height_field!

import Oceananigans.TimeSteppers: update_state!

"""
    update_state!(model::ShallowWaterModel)

Fill halo regions for `model.solution` and `model.tracers`.
"""
function update_state!(model::ShallowWaterModel)

    # Mask immersed fields
    masking_events = Tuple(mask_immersed_field!(field) for field in merge((uh=model.solution.uh, vh=model.solution.vh), model.tracers))

    wait(device(model.architecture), MultiEvent(tuple(masking_events..., height_masking_event)))

    # Fill halos for velocities and tracers
    fill_halo_regions!(merge(model.solution, model.tracers),
                       model.architecture,
                       model.clock,
                       fields(model))

    return nothing
end

