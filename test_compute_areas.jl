using Oceananigans
using Oceananigans.Operators: Δyᶠᶜᵃ, Δxᶜᶠᵃ, Δzᵃᵃᶜ
using CUDA
using Statistics: dot
using BenchmarkTools

function compute_vertically_integrated_lateral_areas!(∫ᶻ_Ax, ∫ᶻ_Ay , grid)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    @inbounds begin
        ∫ᶻ_Ax[i, j] = 0
        ∫ᶻ_Ay[i, j] = 0

        for k in 1:grid.Nz
            ∫ᶻ_Ax[i, j] += Δyᶠᶜᵃ(i, j, k, grid) * Δzᵃᵃᶜ(i, j, k, grid)
            ∫ᶻ_Ay[i, j] += Δxᶜᶠᵃ(i, j, k, grid) * Δzᵃᵃᶜ(i, j, k, grid)
        end
    end
end

function compute_areas2(grid)

    FT = eltype(grid)

    block = (Int(16), Int(16))
    blockgrid  = Int.(((grid.Nx, grid.Ny) ./ block))

    wrk1   = CuArray{FT}(undef, (grid.Nx, grid.Ny)) 
    wrk2   = CuArray{FT}(undef, (grid.Nx, grid.Ny)) 
    
    @cuda threads=block blocks=blockgrid compute_vertically_integrated_lateral_areas!(wrk1, wrk2, grid)

    return wrk1, wrk2
end

function compute_vertical_areas_reduce_fashion!(::Val{FT}, ∫ᶻ_Ax, ∫ᶻ_Ay , grid, ::Val{totSize}, ::Val{block}) where {FT, totSize, block}

	tix   = threadIdx().x 
	bix   = blockIdx().x 
	gdim  = gridDim().x * block
	
    j = bix ÷ grid.Nx + 1
    i = bix - (j - 1) * grid.Nx 
	k = tix + (bix -1) * block
	
    sum1 = FT(0.0)
    sum2 = FT(0.0)
    for t = k:gdim:totSize
        sum1 += Δyᶠᶜᵃ(i, j, t, grid) * Δzᵃᵃᶜ(i, j, t, grid)
        sum2 += Δxᶜᶠᵃ(i, j, t, grid) * Δzᵃᵃᶜ(i, j, t, grid)
    end
    
    shArr1 = @cuStaticSharedMem(FT, block)
	shArr2 = @cuStaticSharedMem(FT, block)
	shArr1[tix] = sum1;
    shArr2[tix] = sum2;

    sync_threads()

	iter = block ÷ 2
    while iter > 0
		if tix < iter + 1
			shArr1[tix] += shArr1[tix+iter]
			shArr2[tix] += shArr2[tix+iter]
        end
        sync_threads()
        iter = iter ÷ 2
	end
	if tix == 1 
        ∫ᶻ_Ax[bix] = shArr1[1]
        ∫ᶻ_Ay[bix] = shArr2[1]
    end

    sync_threads()
end

function compute_areas1(grid)

    FT = eltype(grid)
    
    block      = grid.Nz
    blockgrid  = grid.Ny*grid.Nz

    wrk1   = CuArray{FT}(undef, blockgrid) 
    wrk2   = CuArray{FT}(undef, blockgrid) 
	block = 2^floor(Int, log(2, block-1))

    @cuda threads=block blocks=blockgrid compute_vertical_areas_reduce_fashion!(Val(FT), wrk1, wrk2, grid, Val(blockgrid*block*2), Val(block))

    return wrk1, wrk2
end

N = 256
grid = RectilinearGrid(GPU(), extent=(1, 1, 1), size=(N, N, 32), halo=(1, 1, 1), topology=(Periodic, Periodic, Bounded))

@benchmark compute_areas1(grid)
@benchmark compute_areas2(grid)
