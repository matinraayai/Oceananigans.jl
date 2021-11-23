push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Logging
using MPI
using JLD2
using BenchmarkTools

using Oceananigans
using Oceananigans.Distributed
using Benchmarks
using CUDA

Logging.global_logger(OceananigansLogger())

MPI.Init()

      comm = MPI.COMM_WORLD
local_rank = MPI.Comm_rank(comm)
         R = MPI.Comm_size(comm)

 #assigns one GPU per rank, could increase efficiency but must have enough GPUs
 #CUDA.device!(local_rank)

 Nx = Ny = 1024; Nz = 32
 Rx = Rz = 1
 Ry = R

@assert Rx * Ry * Rz == R

@info "Setting up distributed nonhydrostatic model with N=($Nx, $Ny, $Nz) grid points and ranks=($Rx, $Ry, $Rz) on rank $local_rank..."


Architectures = has_cuda() ? [GPU()] : [CPU()]
for child_arch in Architectures 
    child_arch = GPU()
    topo = (Periodic, Periodic, Bounded)
    arch = MultiArch(child_arch, ranks=(Rx, Ry, Rz))
    grid = RectilinearGrid(arch, topology=topo, size=(Nx, Ny, Nz), extent=(1, 1, 1))
    model = HydrostaticFreeSurfaceModel(grid=grid)

    @info "Warming up distributed nonhydrostatic model on rank $local_rank..."

    time_step!(model, 1) # warmup

    @info "Benchmarking distributed nonhydrostatic model on rank $local_rank..."

    MPI.Barrier(comm)

    trial = @benchmark begin
        time_step!($model, 1)
    end samples=10 evals=1

    MPI.Barrier(comm)

    t_median = BenchmarkTools.prettytime(median(trial).time)
    @info "Done benchmarking on rank $(local_rank). Median time: $t_median"

    jldopen("distributed_nonhydrostatic_model_$(R)ranks_$local_rank.jld2", "w") do file
        file["trial"] = trial
    end
end