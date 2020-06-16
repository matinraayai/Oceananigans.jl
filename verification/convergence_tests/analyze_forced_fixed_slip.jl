using PyPlot, Glob, Printf

include("ConvergenceTests/ConvergenceTests.jl")

defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
removespine(side) = gca().spines[side].set_visible(false)
removespines(sides...) = [removespine(side) for side in sides]

u = ConvergenceTests.ForcedFlowFixedSlip.u

filenameses = [
             glob("data/forced_fixed_slip_xz*Δt6.0e-06.jld2"),
             glob("data/forced_fixed_slip_xz*Δt6.0e-07.jld2"),
             glob("data/forced_fixed_slip_xz*Δt6.0e-08.jld2"),
            ]

Δt = [
      6.0e-06,
      6.0e-07,
      6.0e-08,
     ]

close("all")
fig, ax = subplots()

for (j, filenames) in enumerate(filenameses)

    errors = ConvergenceTests.compute_errors((x, y, z, t) -> u(x, y, t), filenames...)

    sizes = ConvergenceTests.extract_sizes(filenames...)

    Nx = map(sz -> sz[1], sizes)

    names = (L"(x, y)",)
    errors = (errors,)


    for i = 1:length(errors)

        @show error = errors[i]
        name = @sprintf("%s, \$ \\Delta t \$ = %.0e", names[i], Δt[j])

        L₁ = map(err -> err.L₁, error)
        L∞ = map(err -> err.L∞, error)

        @show size(L₁) size(Nx)

        loglog(Nx, L₁, color=defaultcolors[i + j - 1], alpha=0.6, mfc="None",
               linestyle="None", marker="o", label="\$L_1\$-norm, $name")

        loglog(Nx, L∞, color=defaultcolors[i + j - 1], alpha=0.6, mfc="None",
               linestyle="None", marker="^", label="\$L_\\infty\$-norm, $name")
    end

    if j == length(filenameses)
        L₁ = map(err -> err.L₁, errors[1])
        loglog(Nx, L₁[end] * (Nx[end] ./ Nx).^2, "k-", linewidth=1, alpha=0.6, label=L"\sim N_x^{-2}")
    end
end


legend()
xlabel(L"N_x")
ylabel("Norms of the absolute error, \$ | u_{\\mathrm{sim}} - u_{\\mathrm{exact}} | \$")
removespines("top", "right")
title("Convergence for forced fixed slip")
xticks(sort(Nx), ["\$ 2^{$(round(Int, log2(n)))} \$" for n in sort(Nx)])

savefig("figs/forced_fixed_slip_convergence.png", dpi=480)


