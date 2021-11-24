
using Oceananigans
using CUDA
using Statistics: dot
using BenchmarkTools

function reduce_multiply_one_block!(FT, aout, ain, bin, arraySize, ::Val{N}) where N

	bdim  = blockDim().x
	tix   = threadIdx().x 
	bix   = blockIdx().x 
	gdim  = gridDim().x*blockDim().x

	glb   = tix + (bix -1) * bdim
	
    sum = FT(0.0)
    for i = glb:gdim:arraySize
        sum += ain[i] * bin[i]
    end
    
    shArr = @cuStaticSharedMem(FT, N)
	shArr[tix] = sum;

    sync_threads()

	iter = bdim ÷ 2
    while iter > 0
		if tix < iter + 1
			shArr[tix] += shArr[tix+iter]
        end
        sync_threads()
        iter = iter ÷ 2
	end
	if tix == 1 
        aout[bix] = shArr[1]
    end

    sync_threads()
end

function reduce_one_block!(FT, aout, ain, arraySize, ::Val{N}) where N
    
	bdim  = blockDim().x
	tix   = threadIdx().x 
	bix   = blockIdx().x 
	gdim  = gridDim().x*blockDim().x

	glb   = tix + (bix -1) * bdim
	
    sum = 0.0;
    
    for i = glb:gdim:arraySize
        sum += ain[i] 
    end
    
    shArr = @cuStaticSharedMem(FT, N)
	shArr[tix] = sum;

    sync_threads()
    
    iter = bdim ÷ 2
    while iter > 0
		if tix < iter + 1
			shArr[tix] += shArr[tix+iter]
        end
        sync_threads()
        iter = iter ÷ 2
	end
	if tix == 1 
        aout[bix] = shArr[1]
    end

    sync_threads()
end

function parallel_dot(FT, a::CuArray, b::CuArray, mx, my)

    grid  = my;
    block = mx;

    wrk  = CuArray{FT}(undef, grid) 
    
	block = 2^floor(Int, log(2, block-1))

    @cuda threads=block blocks=grid reduce_multiply_one_block!(FT, wrk, a, b, grid*block*2, Val(block))
    @cuda threads=block blocks=1    reduce_one_block!(FT, wrk, wrk, my, Val(block))

    return wrk
end

cpu_grid = RectilinearGrid(CPU(), extent=(1, 1), size=(1024, 1024), halo=(1, 1), topology=(Periodic, Periodic, Flat))
gpu_grid = RectilinearGrid(GPU(), extent=(1, 1), size=(1024, 1024), halo=(1, 1), topology=(Periodic, Periodic, Flat))

gpu_f1 = CenterField(GPU(), gpu_grid)
gpu_f2 = CenterField(GPU(), gpu_grid)

cpu_f1 = CenterField(CPU(), cpu_grid)
cpu_f2 = CenterField(CPU(), cpu_grid)

cpu_a1 =   Array{Float64}(undef, (1024, 1024))
cpu_a2 =   Array{Float64}(undef, (1024, 1024))

gpu_a1 = CuArray{Float64}(undef, (1024, 1024))
gpu_a2 = CuArray{Float64}(undef, (1024, 1024))

gpu_a3 = CuArray{Float32}(undef, (1024, 1024))
gpu_a4 = CuArray{Float32}(undef, (1024, 1024))

gpu_a6 = CUDA.rand(1024, 1024)
gpu_a7 = CUDA.rand(1024, 1024)

FT = eltype(gpu_a6)

set!(gpu_f1, gpu_a6)
set!(gpu_f2, gpu_a7)

@benchmark a = parallel_dot(FT, gpu_a6, gpu_a7, 1024, 1024)
@benchmark b = dot(gpu_a6, gpu_a7)
@benchmark c = dot(gpu_f1, gpu_f2)

CUDA.allowscalar(true)
@test a[1] == b == c

@info "CPU benchmarking"

# @info "benchmarking fields"
# @benchmark dot(cpu_f1, cpu_f2)
# @info "benchmarking arrays"
# @benchmark dot(cpu_a1, cpu_a2)
# @info "benchmarking fields as arrays"
# @benchmark dot(parent(cpu_f1.data), parent(cpu_f2.data))

# @info "GPU benchmarking"

# @info "benchmarking fields"
# @benchmark dot(gpu_f1, gpu_f2)
# @info "benchmarking arrays"
# @benchmark dot(gpu_a1, gpu_a2)
# @info "benchmarking fields as arrays"
# @benchmark dot(parent(gpu_f1.data), parent(gpu_f2.data))