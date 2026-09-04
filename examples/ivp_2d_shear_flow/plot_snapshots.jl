"""
Plot planes from joint analysis files.

Usage:
    julia plot_snapshots.jl <files>... [--output=<dir>]

NOTE: This script requires HDF5.jl and a plotting package such as CairoMakie.jl.
      Install them with: using Pkg; Pkg.add(["HDF5", "CairoMakie"])
"""

using HDF5
# using CairoMakie  # Uncomment when available

function main(filename, start, count, output)
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot settings
    tasks = ["tracer", "pressure", "vorticity"]
    dpi = 200
    title_func(sim_time) = "t = $(round(sim_time; digits=3))"
    savename_func(write) = "write_$(lpad(write, 6, '0')).png"

    # Plot writes
    h5open(filename, "r") do file
        for index in start:(start + count - 1)
            for (n, task) in enumerate(tasks)
                dset = file["tasks"][task]
                data = read(dset)[:, :, index]  # Julia 1-based indexing
                # Plot using CairoMakie or Plots.jl:
                # fig = Figure()
                # ax = Axis(fig[n, 1]; title=task)
                # heatmap!(ax, data')
            end
            # Add time title
            sim_time = read(file["scales/sim_time"])[index]
            title = title_func(sim_time)
            # Save figure
            write_number = read(file["scales/write_number"])[index]
            savename = savename_func(write_number)
            savepath = joinpath(output, savename)
            # save(savepath, fig; px_per_unit=2)
            println("Would save: $savepath ($title)")
        end
    end
end

# Main entry point
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        println("Usage: julia plot_snapshots.jl <files>... [--output=<dir>]")
        exit(1)
    end

    # Parse arguments
    output = "./frames"
    files = String[]
    for arg in ARGS
        if startswith(arg, "--output=")
            output = arg[length("--output=") + 1:end]
        else
            push!(files, arg)
        end
    end

    mkpath(output)

    for filename in files
        h5open(filename, "r") do file
            count = length(read(file["scales/write_number"]))
            main(filename, 1, count, output)
        end
    end
end
