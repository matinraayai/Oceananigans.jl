using CUDA
using AMDGPU
using Oceananigans.Architectures: AbstractCPUArchitecture, GPU, AMD

import Base: zeros

zeros(FT, ::AbstractCPUArchitecture, N...) = zeros(FT, N...)
zeros(FT, ::GPU, N...) = CUDA.zeros(FT, N...)
zeros(FT, ::AMD, N...) = zeros(FT, N...) |> ROCArray # AMDGPU.zeros(FT, N...)

zeros(arch::AbstractArchitecture, grid, N...) = zeros(eltype(grid), arch, N...)
