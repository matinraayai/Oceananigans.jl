# test halo passing
using Oceananigans
using Oceananigans.Distributed
using Oceananigans.BoundaryConditions
using BenchmarkTools
using Test

using MPI
MPI.Init()

R = MPI.Comm_size(MPI.COMM_WORLD)
r = MPI.Comm_rank(MPI.COMM_WORLD) + 1

cpu_arch = MultiArch(CPU(), ranks = (1, R, 1))
gpu_arch = MultiArch(GPU(), ranks = (1, R, 1))

cpu_grid = RectilinearGrid(cpu_arch, extent=(2, 8), size=(2, 8), halo=(1, 1), topology=(Periodic, Periodic, Flat))
gpu_grid = RectilinearGrid(cpu_arch, extent=(2, 8), size=(2, 8), halo=(1, 1), topology=(Periodic, Periodic, Flat))

cpu_arch = cpu_grid.architecture
gpu_arch = gpu_grid.architecture

cpu_field = CenterField(cpu_arch, cpu_grid)
gpu_field = CenterField(gpu_arch, gpu_grid)

set!(cpu_field, r)
set!(gpu_field, r)

fill_halo_regions!(cpu_field, cpu_arch); 
@test all(parent(cpu_field.data)[:,1,:] .== cpu_arch.connectivity.south + 1)
@test all(parent(cpu_field.data)[:,end,:] .== cpu_arch.connectivity.north + 1)
fill_halo_regions!(gpu_field, gpu_arch); 
@test all(Array(parent(gpu_field.data))[:,1,:] .== gpu_arch.connectivity.south + 1)
@test all(Array(parent(gpu_field.data))[:,end,:] .== gpu_arch.connectivity.north + 1)


cpu_trial = @benchmark begin
    fill_halo_regions!($cpu_field, $cpu_arch)
end samples=10 evals=10

gpu_trial = @benchmark begin
    fill_halo_regions!($gpu_field, $gpu_arch)
end samples=10 evals=10

