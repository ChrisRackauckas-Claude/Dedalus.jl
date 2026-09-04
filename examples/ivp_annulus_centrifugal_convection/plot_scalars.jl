"""
Plot scalars.

Usage:
    julia plot_scalars.jl <file>

NOTE: This script requires HDF5.jl and a plotting package such as CairoMakie or Plots.jl.
      Install them with: using Pkg; Pkg.add(["HDF5", "CairoMakie"])
"""

using HDF5
# using CairoMakie  # Uncomment when available

# Parameters
tasks = ["KE"]
figsize = (600, 400)
log_scale = true

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        println("Usage: julia plot_scalars.jl <file>")
        exit(1)
    end

    filename = ARGS[1]
    h5open(filename, "r") do file
        t = read(file["scales/sim_time"])
        for task in tasks
            dset = file["tasks"][task]
            data = vec(read(dset))
            # Plot using CairoMakie:
            # fig = Figure(; size=figsize)
            # ax = Axis(fig[1, 1]; xlabel="t", yscale=log_scale ? log10 : identity)
            # lines!(ax, t, data; label=task)
            # axislegend(ax)
            # save("scalars.pdf", fig)
            println("Task $task: $(length(data)) data points, t=[$(t[1]), $(t[end])]")
        end
    end
end
