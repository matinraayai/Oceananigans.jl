using Oceananigans.Grids
import Oceananigans.Grids: new_data
import Oceananigans.Grids: total_size
using Oceananigans.Grids: R_Earth
import Oceananigans.Grids: xnode
import Oceananigans.Grids: ynode
import Oceananigans.Grids: znode
using Oceananigans.CubedSpheres
using Oceananigans.ImmersedBoundaries
import Oceananigans.ImmersedBoundaries.ImmersedBoundaryGrid
import Oceananigans.ImmersedBoundaries.GridFittedBoundary

import Oceananigans.CubedSpheres: default_face_connectivity
import Oceananigans.CubedSpheres: fill_grid_metric_halos!

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

function ImmersedBoundaryConformalCubedSphereGrid(filepath::AbstractString, FT=Float64; Nz, z, ibg_solid_func, architecture = CPU(), radius = R_Earth, halo = (1, 1, 1))
    @warn "ImmersedBoundaryConformalCubedSphereGrid is super experimental: use with extra caution!"

    face_topo = (Connected, Connected, Bounded)
    halo = (2, 2, 1)
    face_kwargs = (Nz=Nz, z=z, topology=face_topo, radius=radius, halo=halo, architecture=architecture)

    underlying_faces = Tuple(ConformalCubedSphereFaceGrid(filepath, FT; face=n, face_kwargs...) for n in 1:6)
    # CNH with Immersed    : faces = Tuple( ImmersedBoundaryGrid(underlying_faces[n], GridFittedBoundary(ibg_solid_func))  for n in 1:6) 
    # CNH without Immersed : faces = underlying_faces
    faces = Tuple( ImmersedBoundaryGrid(underlying_faces[n], GridFittedBoundary(ibg_solid_func))  for n in 1:6) 

    face_connectivity = default_face_connectivity()

    grid = ConformalCubedSphereGrid{FT, typeof(faces), typeof(face_connectivity)}(faces, face_connectivity)

    fill_grid_metric_halos!(grid)
    fill_grid_metric_halos!(grid)

    return grid
end

@inline xnode(LX, LY, LZ, i, j, k, ibg::ImmersedBoundaryGrid) = xnode(LX, LY, LZ, i, j, k, ibg.grid)
@inline ynode(LX, LY, LZ, i, j, k, ibg::ImmersedBoundaryGrid) = ynode(LX, LY, LZ, i, j, k, ibg.grid)
@inline znode(LX, LY, LZ, i, j, k, ibg::ImmersedBoundaryGrid) = znode(LX, LY, LZ, i, j, k, ibg.grid)
