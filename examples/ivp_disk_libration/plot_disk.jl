"""
Plot disk outputs.

Usage:
    julia plot_disk.jl <files>... [--output=<dir>]

NOTE: This script requires HDF5.jl and a plotting package such as CairoMakie.jl.
      Install them with: using Pkg; Pkg.add(["HDF5", "CairoMakie"])
"""

using HDF5
# using CairoMakie  # Uncomment when available

function main(filename, start, count, output)
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot settings
    tasks = ["vorticity"]
    savename_func(write) = "write_$(lpad(write, 6, '0')).png"
    title_func(sim_time) = "t = $(round(sim_time; digits=3))"
    dpi = 200
    coord_func(phi, r, data) = (r .* cos.(phi), r .* sin.(phi), data)

    # Plotting loop
    h5open(filename, "r") do file
        for index in start:(start + count - 1)
            for (n, task) in enumerate(tasks)
                dset = file["tasks"][task]
                data = read(dset)[:, :, index]  # Julia 1-based indexing

                # To plot in polar coordinates, convert to Cartesian:
                # phi_grid = read(dset.dims[1][1])  # Adjust for HDF5 dim scales
                # r_grid = read(dset.dims[2][1])
                # x, y, z_data = coord_func(phi_grid, r_grid, data)
                # Use heatmap! or surface! to plot
            end
            # Save figure
            sim_time = read(file["scales/sim_time"])[index]
            title = title_func(sim_time)
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
        println("Usage: julia plot_disk.jl <files>... [--output=<dir>]")
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
