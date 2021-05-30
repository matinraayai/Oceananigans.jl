using Oceananigans.Grids
import Oceananigans.Grids: new_data
import Oceananigans.Grids: total_size
using Oceananigans.CubedSpheres
using Oceananigans.ImmersedBoundaries
import Oceananigans.ImmersedBoundaries.ImmersedBoundaryGrid

function new_data(FT, arch, ibg::ImmersedBoundaryGrid{F,B,B,B,T}, loc ) where {T<:ConformalCubedSphereGrid,B<:Any,F<:Real}
         return new_data(FT, arch, ibg.grid, loc )
end     

import Oceananigans.Fields: validate_field_data

function validate_field_data(X, Y, Z, data, ibg::ImmersedBoundaryGrid{F,B,B,B,T} ) where {T<:ConformalCubedSphereGrid,B<:Any,F<:Real}
   validate_field_data(X, Y, Z, data, ibg.grid)
end

import Oceananigans.CubedSpheres: faces
import Oceananigans.CubedSpheres: get_face

function faces(field::Oceananigans.Fields.ReducedField{<:Any,<:Any,<:Any,<:Any,<:Any,<:ImmersedBoundaryGrid,<:Any,<:Any,<:Any})
        println("Get faces for an IBG grid cube sphere field.")
        println("Field type is ",typeof(field))
        Tuple(get_face(field, face_index) for face_index in 1:length(field.data.faces))
end

function get_face(field::ImmersedBoundaryGrid, face_index)
        println("Extracting grid from IBG wrapper")
        get_face(field.grid,face_index)
end
