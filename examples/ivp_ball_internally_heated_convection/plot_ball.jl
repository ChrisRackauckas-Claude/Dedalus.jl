"""
Plot cutaway ball outputs.

Usage:
    julia plot_ball.jl <files>... [--output=<dir>]

NOTE: This script requires HDF5.jl and a plotting package such as CairoMakie.jl.
      For 3D ball rendering, GLMakie may provide better performance.
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
    # r: outer radial face
    # s: starting meridional face
    # e: ending meridional face
    task_r = "T(r=1)"
    task_s = "T(phi=0)"
    task_e = "T(phi=3/2*pi)"
    R = 1
    phis = 0
    phie = 3*pi/2
    dpi = 100
    savename_func(write) = "write_$(lpad(write, 6, '0')).png"

    # Plot writes
    h5open(filename, "r") do file
        dset_o = file["tasks"][task_r]
        dset_s = file["tasks"][task_s]
        dset_e = file["tasks"][task_e]

        # In a full implementation:
        # 1. Read coordinate arrays from dimension scales
        # 2. Build vertex arrays for the cutaway ball
        # 3. Construct outer surface + two meridional cross-sections
        # 4. Color by temperature data

        for index in start:(start + count - 1)
            # Read data for outer surface and meridional cuts
            # data_o = read(dset_o)[:, :, 1, index]
            # data_s = read(dset_s)[1, :, :, index]
            # data_e = read(dset_e)[1, :, :, index]

            # Normalize by mean and std of meridional data
            # mean_val = mean(vcat(vec(data_s), vec(data_e)))
            # std_val = std(vcat(vec(data_s), vec(data_e)))

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
        println("Usage: julia plot_ball.jl <files>... [--output=<dir>]")
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
