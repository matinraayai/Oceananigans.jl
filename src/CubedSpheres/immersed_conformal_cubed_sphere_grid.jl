import Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid

function ImmersedBoundaryGrid(grid::ConformalCubedSphereGrid, immersed_boundary)
    faces = Tuple(ImmersedBoundaryGrid(get_face(grid, i), immersed_boundary) for i = 1:6)
    FT = eltype(grid)
    face_connectivity = grid.face_connectivity

    cubed_sphere_immersed_grid = ConformalCubedSphereGrid{FT, typeof(faces), typeof(face_connectivity)}(faces, face_connectivity)
     
    return cubed_sphere_immersed_grid 
end

import Oceananigans.Operators: Γᶠᶠᵃ

@inline function Γᶠᶠᵃ(i, j, k, ibg::ImmersedBoundaryGrid{F, TX, TY, TZ, G, I}, u, v) where {F,TX,TY,TZ,G<:ConformalCubedSphereFaceGrid,I}
    Γᶠᶠᵃ(i, j, k, ibg.grid, u, v)
end

import Oceananigans.Models.HydrostaticFreeSurfaceModels: mask_immersed_field!
using Oceananigans.Fields: AbstractField
using  Base.Iterators

import Oceananigans.CubedSpheres: get_face
using Oceananigans.Fields: architecture
using Oceananigans.Architectures: device_event
using Oceananigans.Utils: launch!
import Oceananigans.ImmersedBoundaries: _mask_immersed_field!

# Specialized mask dispatch for Cube sphere that iterates over each patch
# AND passes "face"number" arg down to masking kernels
function mask_immersed_field!(field::AbstractField{LX, LY, LZ}, cs_grid::ConformalCubedSphereGrid{FT, NTuple{NT,IG}}) where {LX, LY, LZ, FT, NT, IG<:ImmersedBoundaryGrid}
 function do_single_patch_masking_of_field!(field, patch_grid, face_number)
   field_patch=get_face(field,face_number)
   return mask_immersed_field!(field_patch, face_number, patch_grid)
 end
 rvec = Tuple( do_single_patch_masking_of_field!(field, grid_face, face_index)
               for (face_index, grid_face) in enumerate(cs_grid.faces)
             )
 return rvec
end

function mask_immersed_field!(field::AbstractField{LX, LY, LZ}, face_number, grid::ImmersedBoundaryGrid) where {LX, LY, LZ}
    arch = architecture(field)
    loc = (LX(), LY(), LZ(), face_number)
    # loc = (LX(), LY(), LZ() )
    return launch!(arch, grid, :xyz, _mask_immersed_field!, field, loc, grid; dependencies = device_event(arch))
end

import KernelAbstractions: MultiEvent

function MultiEvent( event::NTuple{N,KAE} ) where {N,KAE}
  mevent=Tuple( [flatten( [event[i] for i in length(event)])...] )
  MultiEvent( mevent )
end


