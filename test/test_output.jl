"""Tests for HDF5 output -- Cartesian only."""

using Test
using Dedalus

@testset "Output" begin

    dealias_range = [1.5]
    dtype_range = [Float64, ComplexF64]
    layout_range = ["g", "c"]
    scales_range = [1, 1.5]
    parallel_range = ["gather", "virtual"]

    @testset "Cartesian output N=$N dealias=$dealias T=$T scales=$output_scales layout=$output_layout parallel=$parallel" for
            N in [8],
            dealias in dealias_range,
            T in dtype_range,
            output_scales in scales_range,
            output_layout in layout_range,
            parallel in parallel_range
        # Bases
        c = CartesianCoordinates("x", "y", "z")
        d = Distributor(c, dtype=T)
        xb = Fourier(c.coords[1], size=N, bounds=(0, 1), dealias=dealias, dtype=T)
        yb = Fourier(c.coords[2], size=N, bounds=(0, 1), dealias=dealias, dtype=T)
        zb = Chebyshev(c.coords[3], size=N, bounds=(0, 1), dealias=dealias)
        x, y, z = local_grids(d, xb, yb, zb)
        # Fields
        u = Field(d, name="u", bases=(xb, yb, zb))
        u["g"] = @. sin(x) * sin(y) * sin(z)
        # Problem
        problem = IVP([u])
        add_equation!(problem, "dt(u) = 0")
        solver = build_solver(problem, "RK111")
        # Output
        tasks = [u, u(x=0), u(y=0), u(z=0),
                 u(x=0, y=0), u(x=0, z=0), u(y=0, z=0),
                 u(x=0, y=0, z=0)]
        tempdir = mktempdir()
        try
            output = add_file_handler(solver.evaluator, tempdir, iter=1, parallel=parallel)
            for task in tasks
                add_task!(output, task, layout=output_layout, name=string(task), scales=output_scales)
            end
            evaluate_handlers!(solver, [output])
            # Check solution by reading HDF5
            h5_path = joinpath(tempdir, basename(tempdir) * "_s1.h5")
            h5open(h5_path, "r") do file
                for task in tasks
                    task_saved = file["tasks"][string(task)][end]
                    task_eval = evaluate(task)
                    change_scales!(task_eval, output_scales)
                    @test isapprox(task_eval[output_layout], task_saved, atol=1e-12)
                end
            end
        finally
            rm(tempdir, recursive=true, force=true)
        end
    end

    @testset "Cartesian output 1D N=$N T=$T" for
            N in [16],
            T in dtype_range
        c = CartesianCoordinates("x")
        d = Distributor(c, dtype=T)
        xb = Fourier(c.coords[1], size=N, bounds=(0, 2 * pi), dealias=1.5, dtype=T)
        x = local_grid(d, xb)
        u = Field(d, name="u", bases=(xb,))
        u["g"] = sin.(x)
        problem = IVP([u])
        add_equation!(problem, "dt(u) = 0")
        solver = build_solver(problem, "RK111")
        tempdir = mktempdir()
        try
            output = add_file_handler(solver.evaluator, tempdir, iter=1, parallel="gather")
            add_task!(output, u, layout="g", name="u", scales=1)
            evaluate_handlers!(solver, [output])
            h5_path = joinpath(tempdir, basename(tempdir) * "_s1.h5")
            h5open(h5_path, "r") do file
                task_saved = file["tasks"]["u"][end]
                u_eval = evaluate(u)
                change_scales!(u_eval, 1)
                @test isapprox(u_eval["g"], task_saved, atol=1e-12)
            end
        finally
            rm(tempdir, recursive=true, force=true)
        end
    end

end
