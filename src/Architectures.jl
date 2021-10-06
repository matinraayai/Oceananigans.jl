module Architectures

export AbstractArchitecture, AbstractCPUArchitecture, AbstractGPUArchitecture
export CPU, GPU, AMD
export device, device_event, architecture, array_type, arch_array

using KernelAbstractions
using CUDA
using CUDAKernels
using AMDGPU
using ROCKernels
using Adapt
using OffsetArrays

"""
    AbstractArchitecture

Abstract supertype for architectures supported by Oceananigans.
"""
abstract type AbstractArchitecture end


"""
    AbstractCPUArchitecture

Abstract supertype for CPU architectures supported by Oceananigans.
"""
abstract type AbstractCPUArchitecture <: AbstractArchitecture end

"""
    AbstractGPUArchitecture

Abstract supertype for GPU architectures supported by Oceananigans.
"""
abstract type AbstractGPUArchitecture <: AbstractArchitecture end

"""
    CPU <: AbstractArchitecture

Run Oceananigans on one CPU node. Uses multiple threads if the environment
variable `JULIA_NUM_THREADS` is set.
"""
struct CPU <: AbstractCPUArchitecture end

"""
    GPU <: AbstractGPUArchitecture

Run Oceananigans on a single NVIDIA CUDA GPU.
"""
struct GPU <: AbstractGPUArchitecture end

"""
    AMD <: AbstractGPUArchitecture

Run Oceananigans on a single AMD ROCm GPU.
"""
struct AMD <: AbstractGPUArchitecture end

device(::AbstractCPUArchitecture) = KernelAbstractions.CPU()
device(::GPU) = CUDAKernels.CUDADevice()
device(::AMD) = ROCKernels.ROCDevice()

architecture() = nothing
architecture(::Number) = nothing
architecture(::Array) = CPU()
architecture(::CuArray) = GPU()
architecture(::ROCArray) = AMD()

array_type(::CPU) = Array
array_type(::GPU) = CuArray
array_type(::AMD) = ROCArray

# No conversion needed if array belongs on the architectrure.
arch_array(::AbstractCPUArchitecture, A::Array) = A
arch_array(::GPU, A::CuArray) = A
arch_array(::AMD, A::ROCArray) = A

# Convert arrays that don't belong on the architecture.
arch_array(::AbstractCPUArchitecture, A) = Array(A)
arch_array(::GPU, A) = CuArray(A)
arch_array(::AMD, A) = ROCArray(A)

const OffsetCPUArray = OffsetArray{FT, N, <:Array} where {FT, N}
const OffsetGPUArray = OffsetArray{FT, N, <:CuArray} where {FT, N}
const OffsetAMDArray = OffsetArray{FT, N, <:ROCArray} where {FT, N}

Adapt.adapt_structure(::CPU, a::OffsetCPUArray) = a
Adapt.adapt_structure(::GPU, a::OffsetGPUArray) = a
Adapt.adapt_structure(::AMD, a::OffsetAMDArray) = a

Adapt.adapt_structure(::CPU, a::OffsetGPUArray) = OffsetArray(Array(a.parent), a.offsets...)
Adapt.adapt_structure(::GPU, a::OffsetCPUArray) = OffsetArray(CuArray(a.parent), a.offsets...)
Adapt.adapt_structure(::AMD, a::OffsetCPUArray) = OffsetArray(ROCArray(a.parent), a.offsets...)

device_event(arch) = Event(device(arch))

# Hacky temporary fix!
# See: https://github.com/JuliaGPU/KernelAbstractions.jl/pull/257
#      https://github.com/JuliaGPU/KernelAbstractions.jl/issues/267

import KernelAbstractions: Event
using ROCKernels: ROCEvent

function Event(::ROCDevice)
    queue = AMDGPU.get_default_queue()
    event = AMDGPU.barrier_and!(queue, AMDGPU.active_kernels(queue))
    MultiEvent(Tuple(ROCEvent(s) for s in event.signals))
end

end
