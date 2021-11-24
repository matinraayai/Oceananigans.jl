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

N    = (128, 128, 32)
topo = (Periodic, Periodic, Periodic)
halo = (3, 3, 3)

cpu_grid = RectilinearGrid(cpu_arch, extent=halo, size=N, halo=halo, topology=topo)
gpu_grid = RectilinearGrid(gpu_arch, extent=halo, size=N, halo=halo, topology=topo)

cpu_arch = cpu_grid.architecture
gpu_arch = gpu_grid.architecture

cpu_field = CenterField(cpu_arch, cpu_grid)
gpu_field = CenterField(gpu_arch, gpu_grid)

set!(cpu_field, r)
set!(gpu_field, r)

Hy1 = 1:cpu_grid.Hy
Hy2 = cpu_grid.Ny + cpu_grid.Hy + 1 : cpu_grid.Ny + 2cpu_grid.Hy

fill_halo_regions!(cpu_field, cpu_arch)

if topo[2] != Periodic
    if r == 1
        @test all(parent(cpu_field.data)[:,Hy2,:] .== cpu_arch.connectivity.north + 1)
    elseif r == R
        @test all(parent(cpu_field.data)[:,Hy1,:] .== cpu_arch.connectivity.south + 1)
    else
        @test all(parent(cpu_field.data)[:,Hy1,:] .== cpu_arch.connectivity.south + 1)
        @test all(parent(cpu_field.data)[:,Hy2,:] .== cpu_arch.connectivity.north + 1)
    end
else
    @test all(parent(cpu_field.data)[:,Hy1,:] .== cpu_arch.connectivity.south + 1)
    @test all(parent(cpu_field.data)[:,Hy2,:] .== cpu_arch.connectivity.north + 1)
end

fill_halo_regions!(gpu_field, gpu_arch)
if topo[2] != Periodic
    if r == 1
        @test all(Array(parent(cpu_field.data))[:,Hy2,:] .== gpu_arch.connectivity.north + 1)
    elseif r == R
        @test all(Array(parent(cpu_field.data))[:,Hy1,:] .== gpu_arch.connectivity.south + 1)
    else
        @test all(Array(parent(cpu_field.data))[:,Hy1,:] .== gpu_arch.connectivity.south + 1)
        @test all(Array(parent(cpu_field.data))[:,Hy2,:] .== gpu_arch.connectivity.north + 1)
    end
else
    @test all(Array(parent(cpu_field.data))[:,Hy1,:] .== gpu_arch.connectivity.south + 1)
    @test all(Array(parent(cpu_field.data))[:,Hy2,:] .== gpu_arch.connectivity.north + 1)
end

cpu_trial = @benchmark begin
    fill_halo_regions!($cpu_field, $cpu_arch)
end samples=10 evals=10

gpu_trial = @benchmark begin
    fill_halo_regions!($gpu_field, $gpu_arch)
end samples=10 evals=10
