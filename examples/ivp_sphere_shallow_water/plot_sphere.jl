"""
Plot sphere outputs.

Usage:
    julia plot_sphere.jl <files>... [--output=<dir>]

NOTE: This script requires HDF5.jl and a plotting package such as CairoMakie.jl.
      For 3D sphere plotting, GLMakie or Meshes.jl may also be useful.
      Install them with: using Pkg; Pkg.add(["HDF5", "CairoMakie"])
"""

using HDF5
# using CairoMakie  # Uncomment when available

function build_s2_coord_vertices(phi, theta)
    phi = vec(phi)
    phi_vert = vcat(phi, [2*pi])
    phi_vert .-= phi_vert[2] / 2
    theta = vec(theta)
    theta_mid = (theta[1:end-1] .+ theta[2:end]) ./ 2
    theta_vert = vcat([pi], theta_mid, [0])
    # Create meshgrid equivalent (ij indexing)
    phi_mesh = repeat(phi_vert, 1, length(theta_vert))
    theta_mesh = repeat(theta_vert', length(phi_vert), 1)
    return phi_mesh, theta_mesh
end

function main(filename, start, count, output)
    """Save plot of specified tasks for given range of analysis writes."""
    # Plot settings
    task = "vorticity"
    dpi = 100
    savename_func(write) = "write_$(lpad(write, 6, '0')).png"

    # Plot writes
    h5open(filename, "r") do file
        dset = file["tasks"][task]
        # Read coordinate data
        # phi = vec(read(dset.dims[1][1]))
        # theta = vec(read(dset.dims[2][1]))
        # phi_vert, theta_vert = build_s2_coord_vertices(phi, theta)
        # x = sin.(theta_vert) .* cos.(phi_vert)
        # y = sin.(theta_vert) .* sin.(phi_vert)
        # z = cos.(theta_vert)

        for index in start:(start + count - 1)
            data = read(dset)[:, :, index]  # Julia 1-based indexing
            clim = maximum(abs.(data))

            # Plot using CairoMakie 3D surface:
            # fig = Figure(; size=(800, 800))
            # ax = Axis3(fig[1, 1])
            # surface!(ax, x, y, z; color=data, colormap=:RdBu, colorrange=(-clim, clim))
            # save(savepath, fig; px_per_unit=1)

            # Save figure
            write_number = read(file["scales/write_number"])[index]
            savename = savename_func(write_number)
            savepath = joinpath(output, savename)
            println("Would save: $savepath")
        end
    end
end

# Main entry point
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        println("Usage: julia plot_sphere.jl <files>... [--output=<dir>]")
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
