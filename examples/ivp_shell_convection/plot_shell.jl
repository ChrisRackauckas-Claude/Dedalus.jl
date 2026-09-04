"""
Plot cutaway spherical shell outputs.

Usage:
    julia plot_shell.jl <files>... [--output=<dir>]

NOTE: This script requires HDF5.jl and a plotting package such as CairoMakie.jl.
      For 3D shell rendering, GLMakie may provide better performance.
      Install them with: using Pkg; Pkg.add(["HDF5", "CairoMakie"])
"""

using HDF5
# using CairoMakie  # Uncomment when available

function build_s2_vertices(phi, theta)
    phi = vec(phi)
    phi_vert = vcat(phi, [2*pi])
    phi_vert .-= phi_vert[2] / 2
    theta = vec(theta)
    theta_mid = (theta[1:end-1] .+ theta[2:end]) ./ 2
    theta_vert = vcat([pi], theta_mid, [0])
    return phi_vert, theta_vert
end

function build_spherical_vertices(phi, theta, r, Ri, Ro)
    phi_vert, theta_vert = build_s2_vertices(phi, theta)
    r = vec(r)
    r_mid = (r[1:end-1] .+ r[2:end]) ./ 2
    r_vert = vcat([Ri], r_mid, [Ro])
    return phi_vert, theta_vert, r_vert
end

function spherical_to_cartesian(phi, theta, r)
    # Create 3D meshgrid (ij indexing)
    np = length(phi)
    nt = length(theta)
    nr = length(r)
    x = zeros(np, nt, nr)
    y = zeros(np, nt, nr)
    z = zeros(np, nt, nr)
    for (ip, p) in enumerate(phi)
        for (it, t) in enumerate(theta)
            for (ir, rv) in enumerate(r)
                x[ip, it, ir] = rv * sin(t) * cos(p)
                y[ip, it, ir] = rv * sin(t) * sin(p)
                z[ip, it, ir] = rv * cos(t)
            end
        end
    end
    return x, y, z
end

function main(filename, start, count, output)
    """Save plot of specified tasks for given range of analysis writes."""
    # Plot settings
    task = "flux"
    Ri, Ro = 14, 15
    phis = 0
    phie = 3*pi/2
    dpi = 100
    savename_func(write) = "write_$(lpad(write, 6, '0')).png"

    # Plot writes
    h5open(filename, "r") do file
        # Read datasets
        dset_i = file["tasks"]["$(task)_r_inner"]
        dset_o = file["tasks"]["$(task)_r_outer"]
        dset_s = file["tasks"]["$(task)_phi_start"]
        dset_e = file["tasks"]["$(task)_phi_end"]

        # In a full implementation, read coordinate arrays from dim scales
        # and construct the 3D cutaway shell visualization

        for index in start:(start + count - 1)
            # Read data slices for inner/outer surfaces and meridional cuts
            # data_i = read(dset_i)[:, :, 1, index]
            # data_o = read(dset_o)[:, :, 1, index]
            # data_s = read(dset_s)[1, :, :, index]
            # data_e = read(dset_e)[1, :, :, index]

            # Construct cutaway shell visualization using 3D surface plots
            # This requires careful assembly of the inner/outer surfaces
            # and meridional cross-sections

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
        println("Usage: julia plot_shell.jl <files>... [--output=<dir>]")
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
