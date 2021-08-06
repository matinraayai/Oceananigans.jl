import Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid

function ImmersedBoundaryGrid(grid::ConformalCubedSphereGrid, immersed_boundary)
    faces = Tuple( ImmersedBoundaryGrid(get_face(grid, i), immersed_boundary[i]) for i = 1:6)   # Note - get_face doesn't actually do what its name implies
                                                                                                # except for a few specific types. Not really sure whether
                                                                                                # its reason for existence makes much sense!
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
##CNH function mask_immersed_field!(field::AbstractField{LX, LY, LZ}, cs_grid::ConformalCubedSphereGrid{FT, NTuple{NT,IG}}, loc) where {LX, LY, LZ, FT, NT, IG<:ImmersedBoundaryGrid}
##CNH  function do_single_patch_masking_of_field!(field, patch_grid, face_number)
##CNH    field_patch=get_face(field,face_number)
##CNH    return mask_immersed_field!(field_patch, face_number, patch_grid)
##CNH  end
##CNH  rvec = Tuple( do_single_patch_masking_of_field!(field, grid_face, face_index)
##CNH                for (face_index, grid_face) in enumerate(cs_grid.faces)
##CNH              )
##CNH  return rvec
##CNH end

# ConformalCubedSphereGrid{FT,TU,NT} where {FT,TU<:Tuple{I1,I2,I3,I4,I5,I6},NT} where {I1<:ImmersedBoundaryGrid,I2,I3,I4,I5,I6}
function mask_immersed_field!(field::AbstractField{LX, LY, LZ}, cs_grid::ConformalCubedSphereGrid{FT, TU}, loc) where {LX, LY, LZ, FT, TU<:Tuple{IG1,IG2,IG3,IG4,IG5,IG6}} where {IG1<:ImmersedBoundaryGrid,IG2,IG3,IG4,IG5,IG6}
  # Have cube sphere 6 faces field, need to mask each one
  function do_single_patch_masking_of_field!(field, patch_grid, face_number)
   field_patch=get_face(field,face_number)
   return mask_immersed_field!(field_patch, face_number, patch_grid)
  end
  rvec = Tuple( do_single_patch_masking_of_field!(field, grid_face, face_index)
                for (face_index, grid_face) in enumerate(cs_grid.faces)
              )
  return rvec
end

## CNH THIS FUNCTION SHOULD NO LONGER BE NEEDED
function mask_immersed_field!(field::AbstractField{LX, LY, LZ}, face_number, grid::ImmersedBoundaryGrid) where {LX, LY, LZ}
    arch = architecture(field)
    loc = (LX(), LY(), LZ(), face_number)
    # loc = (LX(), LY(), LZ() )
    mask_value = zero(eltype(grid))
    return launch!(arch, grid, :xyz, _mask_immersed_field!, field, loc, grid, mask_value; dependencies = device_event(arch))
end

import KernelAbstractions: MultiEvent

function MultiEvent( event::NTuple{N,KAE} ) where {N,KAE}
  mevent=Tuple( [flatten( [event[i] for i in length(event)])...] )
  MultiEvent( mevent )
end


