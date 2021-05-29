using Oceananigans.Grids
import Oceananigans.Grids: new_data
import Oceananigans.Grids: total_size
using Oceananigans.CubedSpheres
using Oceananigans.ImmersedBoundaries
import Oceananigans.ImmersedBoundaries.ImmersedBoundaryGrid

function new_data(FT, arch, ibg::ImmersedBoundaryGrid{F,B,B,B,T}, loc ) where {T<:ConformalCubedSphereGrid,B<:Any,F<:Real}
         return new_data(FT, arch, ibg.grid, loc )
end     

### function total_size(loc, ibg::ImmersedBoundaryGrid{F,B,B,B,T} ) where {T<:ConformalCubedSphereGrid,B<:Any,F<:Real}
###         return total_size(loc, ibg.grid)
### end


import Oceananigans.Fields: validate_field_data

function validate_field_data(X, Y, Z, data, ibg::ImmersedBoundaryGrid{F,B,B,B,T} ) where {T<:ConformalCubedSphereGrid,B<:Any,F<:Real}
   validate_field_data(X, Y, Z, data, ibg.grid)
end

